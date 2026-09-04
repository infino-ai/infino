// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-time term expansion: a caller-supplied vocabulary of **stop
//! terms** (dropped from a query) and **term groups** (a head word plus
//! the surface forms that should count as it), applied to a query where
//! it is parsed. The index is never touched — postings stay keyed by the
//! surface form, so `exact_match`, a literal `token_match` and the SQL
//! `LIKE` pushdown keep working — and a query with no expansion runs
//! exactly the code it runs without this module.
//!
//! [`QueryExpansion`] is the public, analyzer-agnostic value a caller
//! builds. Before it can touch a query it is normalized against the
//! target column's analyzer into a [`NormalizedExpansion`]: every stop
//! term, head and member is run through the column's tokenizer and must
//! come out as exactly one term, so `Running` becomes `running` on an
//! `ascii_lower` column and `New York` is rejected instead of silently
//! matching nothing. The engine keeps the normalized form — registered
//! per column on the table handle, or built for one call — and consults
//! it at the single place a query is parsed.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    mem,
};

use thiserror::Error;

use crate::superfile::fts::tokenize::{ParsedQuery, Tokenizer};

/// A query-time vocabulary applied to `bm25_search`, `token_match` and
/// `count`: stop terms and term groups. Nothing in it knows a language or
/// a stemmer; the caller supplies the words.
///
/// - **Stop terms** are removed from a query's bare tokens before
///   matching. A quoted phrase keeps every word (its adjacency is the
///   caller's intent), and a `+term` or `-term` keeps its sigil and is
///   never dropped — an explicit sigil is the caller overriding the
///   vocabulary. If removing stop terms would leave a query with no bare
///   token, no must and no phrase, the bare tokens are kept unchanged, so
///   `the who` on a music column still searches for `the` and `who`.
/// - **Term groups** score a head and its surface forms as one term. A
///   bare, `+` or `-` query token that equals a group head expands into
///   the whole group: in `or` mode the group is one should, in `and`
///   mode one must satisfied by any member, and a `-group` excludes a
///   document holding any member. A group's `tf` at a document is the
///   sum of its members' frequencies and its `idf` is that of the member
///   with the largest document frequency — what a stemmed index would
///   have stored for the stem, without rebuilding the index. Members are
///   not chased through other groups, and phrase words are not expanded.
///
/// Terms are matched against the **column analyzer's output**: at
/// registration (or per call) every entry is run through the column's
/// tokenizer and must yield exactly one term. `exact_match` never
/// expands. An empty expansion changes nothing.
///
/// ```ignore
/// let vocab = QueryExpansion::new()
///     .stop(["the", "and", "of"])
///     .group("run", ["runs", "running", "ran"]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryExpansion {
    /// Stop terms as supplied, in insertion order.
    stop: Vec<String>,
    /// Groups as supplied: `(head, members)`, in insertion order. The
    /// same head may appear more than once; normalization merges them.
    groups: Vec<(String, Vec<String>)>,
}

impl QueryExpansion {
    /// An empty expansion: no stop terms, no groups.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add stop terms. Each is matched as one analyzer term.
    pub fn stop<I, S>(mut self, terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop.extend(terms.into_iter().map(Into::into));
        self
    }

    /// Add a term group: `head` and the `members` that count as it. A
    /// query token equal to `head` expands to the whole group. Calling
    /// this twice with the same head merges the members.
    pub fn group<H, I, S>(mut self, head: H, members: I) -> Self
    where
        H: Into<String>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.groups
            .push((head.into(), members.into_iter().map(Into::into).collect()));
        self
    }

    /// True when the expansion carries no stop terms and no groups.
    pub fn is_empty(&self) -> bool {
        self.stop.is_empty() && self.groups.is_empty()
    }
}

/// Why a [`QueryExpansion`] entry was refused by a column's analyzer.
/// Surfaces to callers as a configuration / invalid-query error naming
/// the column and the entry.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExpansionError {
    /// A stop term, group head or group member did not come out of the
    /// column's tokenizer as exactly one term. Multi-word entries and
    /// entries the analyzer drops entirely (a non-ASCII word on an
    /// `ascii_lower` column) both land here.
    #[error(
        "expansion entry {entry:?} tokenizes to {tokens:?}; every stop term, group head and \
         group member must be exactly one term under the column's analyzer"
    )]
    NotOneTerm { entry: String, tokens: Vec<String> },
}

/// A [`QueryExpansion`] resolved against one column's analyzer: every
/// entry is a single analyzer term, so query tokens (which come out of
/// the same tokenizer) can be looked up by plain string equality.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NormalizedExpansion {
    stop: HashSet<String>,
    /// Head → the head followed by its members, deduplicated. A group
    /// that reduces to the head alone is not kept: expanding it would
    /// change nothing, and leaving the token a plain term keeps it on
    /// the term-only fast paths.
    groups: HashMap<String, Vec<String>>,
}

impl NormalizedExpansion {
    /// Run every entry of `expansion` through `tokenizer`. Fails on the
    /// first entry that is not exactly one term.
    pub(crate) fn normalize(
        expansion: &QueryExpansion,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Self, ExpansionError> {
        let mut stop = HashSet::with_capacity(expansion.stop.len());
        for entry in &expansion.stop {
            stop.insert(one_term(tokenizer, entry)?);
        }
        let mut groups: HashMap<String, Vec<String>> =
            HashMap::with_capacity(expansion.groups.len());
        for (head, members) in &expansion.groups {
            let head = one_term(tokenizer, head)?;
            let list = groups.entry(head.clone()).or_default();
            if list.is_empty() {
                list.push(head);
            }
            for member in members {
                let member = one_term(tokenizer, member)?;
                if !list.contains(&member) {
                    list.push(member);
                }
            }
        }
        groups.retain(|_, list| list.len() > 1);
        Ok(Self { stop, groups })
    }

    /// True when applying this expansion cannot change any query.
    pub(crate) fn is_empty(&self) -> bool {
        self.stop.is_empty() && self.groups.is_empty()
    }

    /// Whether `term` (an analyzer output) is a stop term.
    fn is_stop(&self, term: &str) -> bool {
        self.stop.contains(term)
    }

    /// The group `term` heads — the head first, then its members — or
    /// `None` when `term` is not a head.
    fn group_for(&self, term: &str) -> Option<&[String]> {
        self.groups.get(term).map(Vec::as_slice)
    }

    /// Apply this expansion to a freshly parsed query, before the default
    /// operator resolves polarity — the one step every search path
    /// (ranked and unranked) shares, so they can never disagree on what a
    /// query means.
    ///
    /// 1. Stop removal touches the **bare** tokens only. A quoted phrase
    ///    keeps every word, and a `+term` / `-term` keeps its sigil: an
    ///    explicit sigil is the caller overriding the vocabulary. If
    ///    removal would leave no bare token, no must and no phrase, the
    ///    bare tokens are kept unchanged — to the caller, "nothing matched
    ///    because every word was a stop term" is indistinguishable from a
    ///    corpus miss, and `the who` on a music column must still find the
    ///    band.
    /// 2. Every remaining bare, `+` and `-` token that heads a group moves
    ///    out of its term list into the matching group list as the head
    ///    plus its members. Members are not chased through other groups,
    ///    and phrase words are never expanded.
    ///
    /// An empty expansion hands `parsed` back untouched, allocating
    /// nothing.
    pub(crate) fn apply<'q>(&self, mut parsed: ParsedQuery<'q>) -> ParsedQuery<'q> {
        if !self.stop.is_empty() && !parsed.positives.is_empty() {
            let mut bare = mem::take(&mut parsed.positives);
            let kept = bare.iter().filter(|t| !self.is_stop(t)).count();
            let nothing_left = kept == 0
                && parsed.musts.is_empty()
                && parsed.must_phrases.is_empty()
                && parsed.positive_phrases.is_empty();
            if !nothing_left {
                bare.retain(|t| !self.is_stop(t));
            }
            parsed.positives = bare;
        }
        if !self.groups.is_empty() {
            self.split_groups(&mut parsed.musts, &mut parsed.must_groups);
            self.split_groups(&mut parsed.positives, &mut parsed.positive_groups);
            self.split_groups(&mut parsed.negatives, &mut parsed.negative_groups);
        }
        parsed
    }

    /// Move every token of `terms` that heads a group into `groups` (as
    /// the head followed by its members); the rest stay in `terms`, in
    /// their original order.
    fn split_groups(&self, terms: &mut Vec<Cow<'_, str>>, groups: &mut Vec<Vec<String>>) {
        if terms.is_empty() {
            return;
        }
        for token in mem::take(terms) {
            match self.group_for(&token) {
                Some(members) => groups.push(members.to_vec()),
                None => terms.push(token),
            }
        }
    }
}

/// Tokenize one vocabulary entry, requiring exactly one term.
fn one_term(tokenizer: &dyn Tokenizer, entry: &str) -> Result<String, ExpansionError> {
    let mut tokens: Vec<String> = tokenizer.tokenize(entry).collect();
    match tokens.len() {
        1 => Ok(tokens.swap_remove(0)),
        _ => Err(ExpansionError::NotOneTerm {
            entry: entry.to_owned(),
            tokens,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::fts::{reader::BoolMode, tokenize::AsciiLowerTokenizer};

    fn normalized(expansion: &QueryExpansion) -> NormalizedExpansion {
        NormalizedExpansion::normalize(expansion, &AsciiLowerTokenizer).expect("normalizes")
    }

    /// The normalized group `head` leads, as owned strings for `assert_eq!`.
    fn group<'a>(norm: &'a NormalizedExpansion, head: &str) -> Option<Vec<&'a str>> {
        norm.groups
            .get(head)
            .map(|list| list.iter().map(String::as_str).collect())
    }

    #[test]
    fn empty_expansion_is_empty_before_and_after_normalization() {
        let exp = QueryExpansion::new();
        assert!(exp.is_empty());
        assert!(normalized(&exp).is_empty());
    }

    #[test]
    fn single_token_entries_are_accepted_and_folded_by_the_analyzer() {
        // Case folding happens through the column's tokenizer, so a
        // caller who passes `Running` gets `running`.
        let exp = QueryExpansion::new()
            .stop(["The", "AND"])
            .group("Run", ["Runs", "RUNNING", "ran"]);
        let norm = normalized(&exp);
        assert!(norm.stop.contains("the"));
        assert!(norm.stop.contains("and"));
        assert!(
            !norm.stop.contains("The"),
            "entries are stored as analyzer output"
        );
        assert_eq!(
            group(&norm, "run"),
            Some(vec!["run", "runs", "running", "ran"])
        );
        assert_eq!(group(&norm, "runs"), None, "members are not heads");
    }

    #[test]
    fn multi_token_entries_are_rejected_with_the_offending_entry() {
        let exp = QueryExpansion::new().group("run", ["New York"]);
        let err = NormalizedExpansion::normalize(&exp, &AsciiLowerTokenizer)
            .expect_err("two tokens must be refused");
        assert_eq!(
            err,
            ExpansionError::NotOneTerm {
                entry: "New York".into(),
                tokens: vec!["new".into(), "york".into()],
            }
        );
        // A stop term and a head are held to the same rule.
        assert!(
            NormalizedExpansion::normalize(
                &QueryExpansion::new().stop(["of the"]),
                &AsciiLowerTokenizer
            )
            .is_err()
        );
        assert!(
            NormalizedExpansion::normalize(
                &QueryExpansion::new().group("new york", ["nyc"]),
                &AsciiLowerTokenizer
            )
            .is_err()
        );
    }

    #[test]
    fn entries_the_analyzer_drops_are_rejected_not_ignored() {
        // `ascii_lower` drops a non-ASCII token entirely, so the entry
        // tokenizes to nothing — a silent no-op would hide the mistake.
        let exp = QueryExpansion::new().stop(["café"]);
        let err = NormalizedExpansion::normalize(&exp, &AsciiLowerTokenizer)
            .expect_err("a dropped token is not one term");
        assert!(matches!(
            err,
            ExpansionError::NotOneTerm { ref entry, ref tokens } if entry == "café" && tokens.is_empty()
        ));
    }

    #[test]
    fn groups_are_deduplicated_and_merged_by_head() {
        // The head listed among its own members, a member repeated in two
        // spellings, and a second `group` call for the same head all
        // collapse into one list with the head first and no repeats.
        let exp = QueryExpansion::new()
            .group("run", ["runs", "run", "Runs"])
            .group("RUN", ["ran", "runs"]);
        let norm = normalized(&exp);
        assert_eq!(group(&norm, "run"), Some(vec!["run", "runs", "ran"]));
    }

    #[test]
    fn a_group_of_only_its_head_is_dropped() {
        // Expanding `run` to `[run]` changes nothing, so the token stays
        // a plain term and the expansion counts as empty.
        let exp = QueryExpansion::new().group("run", ["run", "RUN"]);
        let norm = normalized(&exp);
        assert_eq!(group(&norm, "run"), None);
        assert!(norm.is_empty());
        assert!(!exp.is_empty(), "the raw value still records the call");
    }

    #[test]
    fn stop_terms_are_deduplicated() {
        let exp = QueryExpansion::new().stop(["the", "The", "the"]);
        let norm = normalized(&exp);
        assert_eq!(norm.stop.len(), 1);
        assert!(norm.stop.contains("the"));
    }

    #[test]
    fn builder_accumulates_across_calls() {
        let exp = QueryExpansion::new()
            .stop(["a"])
            .stop(["b"])
            .group("x", ["y"])
            .group("p", ["q"]);
        let norm = normalized(&exp);
        assert!(norm.stop.contains("a") && norm.stop.contains("b"));
        assert!(group(&norm, "x").is_some() && group(&norm, "p").is_some());
    }

    // ---- apply: the parse-point rewrite ----

    /// The test vocabulary: three stop terms and two families.
    fn vocabulary() -> NormalizedExpansion {
        normalized(
            &QueryExpansion::new()
                .stop(["the", "and", "of"])
                .group("run", ["runs", "running", "ran"])
                .group("fail", ["fails", "failing", "failed"]),
        )
    }

    fn parse_and_apply<'q>(norm: &NormalizedExpansion, query: &'q str) -> ParsedQuery<'q> {
        norm.apply(AsciiLowerTokenizer.parse(query))
    }

    fn strs<'a>(tokens: &'a [Cow<'a, str>]) -> Vec<&'a str> {
        tokens.iter().map(|t| t.as_ref()).collect()
    }

    fn run_group() -> Vec<String> {
        vec!["run".into(), "runs".into(), "running".into(), "ran".into()]
    }

    fn fail_group() -> Vec<String> {
        vec![
            "fail".into(),
            "fails".into(),
            "failing".into(),
            "failed".into(),
        ]
    }

    #[test]
    fn stop_removal_touches_bare_tokens_only() {
        let p = parse_and_apply(&vocabulary(), "the login and page +the -of");
        assert_eq!(strs(&p.positives), vec!["login", "page"]);
        // An explicit sigil keeps the term: the caller overrode the
        // vocabulary.
        assert_eq!(strs(&p.musts), vec!["the"]);
        assert_eq!(strs(&p.negatives), vec!["of"]);
    }

    #[test]
    fn phrase_words_are_never_removed_or_expanded() {
        let p = parse_and_apply(&vocabulary(), "\"the who\" \"running fails\" the");
        // The bare `the` goes; the quoted ones stay, unexpanded.
        assert!(p.positives.is_empty());
        assert_eq!(p.positive_phrases.len(), 2);
        assert_eq!(strs(&p.positive_phrases[0]), vec!["the", "who"]);
        assert_eq!(strs(&p.positive_phrases[1]), vec!["running", "fails"]);
        assert!(p.positive_groups.is_empty());
    }

    #[test]
    fn all_stop_query_keeps_its_bare_tokens() {
        // Nothing but stop terms and no other positive atom: dropping them
        // would make the query empty, so it is left alone.
        let p = parse_and_apply(&vocabulary(), "the and of");
        assert_eq!(strs(&p.positives), vec!["the", "and", "of"]);
        // A negative alone does not rescue the query — there is still
        // nothing positive to match, so the bare tokens stay.
        let p = parse_and_apply(&vocabulary(), "the -login");
        assert_eq!(strs(&p.positives), vec!["the"]);
        assert_eq!(strs(&p.negatives), vec!["login"]);
        // A must, a must-phrase or a bare phrase does rescue it: the stop
        // terms go and the remaining atom carries the query.
        let p = parse_and_apply(&vocabulary(), "+login the and");
        assert!(p.positives.is_empty());
        assert_eq!(strs(&p.musts), vec!["login"]);
        let p = parse_and_apply(&vocabulary(), "\"login page\" the");
        assert!(p.positives.is_empty());
        assert_eq!(p.positive_phrases.len(), 1);
        let p = parse_and_apply(&vocabulary(), "+\"login page\" of");
        assert!(p.positives.is_empty());
        assert_eq!(p.must_phrases.len(), 1);
    }

    #[test]
    fn group_heads_move_into_the_matching_group_list() {
        let p = parse_and_apply(&vocabulary(), "+run login fail -running -page");
        assert_eq!(p.must_groups, vec![run_group()]);
        assert!(p.musts.is_empty());
        assert_eq!(strs(&p.positives), vec!["login"]);
        assert_eq!(p.positive_groups, vec![fail_group()]);
        // `running` is a member, not a head: it stays a literal negative.
        assert_eq!(strs(&p.negatives), vec!["running", "page"]);
        assert!(p.negative_groups.is_empty());
    }

    #[test]
    fn members_are_not_chased_and_order_is_preserved() {
        // `runs` heads its own group whose member `ran` is also a member of
        // `run`; expanding `run` yields `run`'s list only — no transitive
        // closure through `runs`.
        let norm = normalized(
            &QueryExpansion::new()
                .group("run", ["runs", "ran"])
                .group("runs", ["ran", "sprint"]),
        );
        let p = parse_and_apply(&norm, "alpha run beta runs gamma");
        assert_eq!(strs(&p.positives), vec!["alpha", "beta", "gamma"]);
        assert_eq!(
            p.positive_groups,
            vec![
                vec!["run".to_string(), "runs".into(), "ran".into()],
                vec!["runs".to_string(), "ran".into(), "sprint".into()],
            ]
        );
    }

    #[test]
    fn a_stop_term_that_is_also_a_head_is_dropped_when_bare_and_expanded_when_sigiled() {
        let norm = normalized(&QueryExpansion::new().stop(["run"]).group("run", ["runs"]));
        let p = parse_and_apply(&norm, "run login");
        assert_eq!(strs(&p.positives), vec!["login"]);
        assert!(p.positive_groups.is_empty());
        let p = parse_and_apply(&norm, "+run login");
        assert_eq!(p.must_groups, vec![vec!["run".to_string(), "runs".into()]]);
    }

    #[test]
    fn an_empty_expansion_leaves_the_parse_unchanged() {
        let norm = normalized(&QueryExpansion::new());
        let query = "+run the \"login page\" -fails";
        let p = parse_and_apply(&norm, query);
        let raw = AsciiLowerTokenizer.parse(query);
        assert_eq!(strs(&p.musts), strs(&raw.musts));
        assert_eq!(strs(&p.positives), strs(&raw.positives));
        assert_eq!(strs(&p.negatives), strs(&raw.negatives));
        assert_eq!(p.positive_phrases.len(), raw.positive_phrases.len());
        assert!(p.must_groups.is_empty() && p.positive_groups.is_empty());
        assert!(p.negative_groups.is_empty());
    }

    #[test]
    fn into_clauses_resolves_bare_groups_by_mode() {
        let p = parse_and_apply(&vocabulary(), "run +fail -ran");
        let or = vocabulary()
            .apply(AsciiLowerTokenizer.parse("run +fail -ran"))
            .into_clauses(BoolMode::Or);
        assert_eq!(or.should_groups, vec![run_group()]);
        assert_eq!(or.must_groups, vec![fail_group()]);
        assert_eq!(strs(&or.negatives), vec!["ran"]);
        let and = p.into_clauses(BoolMode::And);
        assert!(and.should_groups.is_empty());
        assert_eq!(and.must_groups, vec![fail_group(), run_group()]);
    }
}
