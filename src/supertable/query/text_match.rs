// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `text_match(column, query [, mode])`: a token match as a `WHERE`
//! predicate.
//!
//! ## Why
//!
//! `token_match` is a table function, so a filter that combines a text
//! match with anything else has to join it back by id:
//!
//! ```sql
//! SELECT * FROM t
//! WHERE ts > '2023-01-03' AND (
//!   _id IN (SELECT _id FROM token_match('title', 'rust'))
//!   OR _id IN (SELECT _id FROM token_match('body', 'rust')))
//! LIMIT 10
//! ```
//!
//! Each subquery collects the id of every matching row before the join can
//! start, so `LIMIT` cannot stop it early, a common term holds every one of
//! its rows' ids in memory at once, and the scan reads every row to probe
//! the join. `text_match` says the same thing as a predicate on the row:
//!
//! ```sql
//! SELECT * FROM t
//! WHERE ts > '2023-01-03' AND (text_match(title, 'rust') OR text_match(body, 'rust'))
//! LIMIT 10
//! ```
//!
//! On a column indexed by `standard`, the scan answers it exactly from the
//! term dictionary and the postings: each query token is one dictionary
//! key, and its postings are the rows (see `candidate::ExactFilter`). The
//! scan then decodes only those rows, intersects them with whatever else
//! the `WHERE` bounds, and stops at the `LIMIT`.
//!
//! ## What it means
//!
//! `text_match(column, query, mode)` is true when the value of `column`,
//! split by the `standard` analyzer, holds every (`'and'`) or any (`'or'`,
//! the default, as `token_match`'s) of the tokens `standard` splits `query`
//! into. `query` is plain words: no phrases or `+` / `-` operators (a quote
//! or a sign is a separator, as in the text). A query with no token matches
//! nothing, and a NULL value is NULL, so `WHERE` drops it.
//!
//! Row by row, the function tokenizes each value with `standard`. The scan
//! uses the postings instead only where they give the same answer: a column
//! whose index was built with `standard`. Under any other analyzer (a
//! stemmer, stopwords) the postings would disagree with the definition, so
//! the scan leaves the predicate to DataFusion, which calls the function on
//! each row.

use std::{collections::HashSet, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BooleanArray, LargeStringArray, StringArray, StringViewArray,
    builder::BooleanBuilder,
};
use arrow_schema::DataType;
use datafusion::{
    common::{ScalarValue, exec_err, plan_err, tree_node::TreeNode},
    error::Result as DfResult,
    execution::context::SessionContext,
    logical_expr::{
        ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        TypeSignature, Volatility, expr::ScalarFunction,
    },
};

use crate::superfile::fts::tokenize::StandardTokenizer;

/// SQL name of the token-match predicate.
pub(crate) const TEXT_MATCH_UDF: &str = "text_match";

/// Whether a value must hold every query token or any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextMatchMode {
    /// Every token (`'and'`).
    All,
    /// Any token (`'or'`, the default).
    Any,
}

impl TextMatchMode {
    /// `'and'` / `'or'`, case-insensitive; `None` for anything else.
    fn parse(mode: &str) -> Option<Self> {
        if mode.eq_ignore_ascii_case("and") {
            Some(TextMatchMode::All)
        } else if mode.eq_ignore_ascii_case("or") {
            Some(TextMatchMode::Any)
        } else {
            None
        }
    }
}

/// The distinct tokens `standard` splits `query` into, in first-seen order.
pub(crate) fn query_tokens(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    StandardTokenizer::new().tokenize_each_inline(query, |token| {
        if seen.insert(token.to_owned()) {
            out.push(token.to_owned());
        }
    });
    out
}

/// A `text_match(column, query [, mode])` call read off a logical
/// expression: its column's name, its query's tokens and its mode. `None`
/// when `expr` is not such a call, or its column is not a bare column, or
/// its query or mode is not a string literal (DataFusion then evaluates
/// it, and the function reports what is wrong).
pub(crate) fn text_match_call(expr: &Expr) -> Option<(String, Vec<String>, TextMatchMode)> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return None;
    };
    if func.name() != TEXT_MATCH_UDF || !(2..=3).contains(&args.len()) {
        return None;
    }
    let Expr::Column(column) = &args[0] else {
        return None;
    };
    let query = string_literal(&args[1])?;
    let mode = match args.get(2) {
        Some(mode) => TextMatchMode::parse(string_literal(mode)?)?,
        None => TextMatchMode::Any,
    };
    Some((column.name.clone(), query_tokens(query), mode))
}

/// Whether `expr` calls `text_match` anywhere in it.
pub(crate) fn holds_text_match(expr: &Expr) -> bool {
    expr.exists(|e| {
        Ok(matches!(
            e,
            Expr::ScalarFunction(ScalarFunction { func, .. }) if func.name() == TEXT_MATCH_UDF
        ))
    })
    .unwrap_or(true)
}

fn string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Literal(
            ScalarValue::Utf8(Some(s))
            | ScalarValue::LargeUtf8(Some(s))
            | ScalarValue::Utf8View(Some(s)),
            _,
        ) => Some(s),
        _ => None,
    }
}

/// The `text_match` scalar function. Stateless: every table registers the
/// same one.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextMatchFunc {
    signature: Signature,
}

impl TextMatchFunc {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
        }
    }
}

/// Register `text_match` on `ctx`.
pub(crate) fn register_text_match(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(TextMatchFunc::new()));
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Null
    )
}

/// A literal argument's string, or an error naming the argument.
fn literal_arg<'a>(value: &'a ColumnarValue, what: &str) -> DfResult<Option<&'a str>> {
    match value {
        ColumnarValue::Scalar(
            ScalarValue::Utf8(s) | ScalarValue::LargeUtf8(s) | ScalarValue::Utf8View(s),
        ) => Ok(s.as_deref()),
        ColumnarValue::Scalar(ScalarValue::Null) => Ok(None),
        _ => exec_err!("{TEXT_MATCH_UDF}: the {what} must be a string literal"),
    }
}

/// Whether `value` holds the `tokens` under `mode`.
fn holds(
    tokenizer: &StandardTokenizer,
    value: &str,
    tokens: &[String],
    mode: TextMatchMode,
    lowered: &mut String,
) -> bool {
    if tokens.is_empty() {
        return false;
    }
    // Every token `standard` takes from ASCII text is that text's piece,
    // lowercased: so a query token missing from the lowercased value as a
    // substring is no token of it, and most values are decided without
    // being tokenized. (Non-ASCII text can lowercase into an ASCII token,
    // the Kelvin sign into `k`, so it is always tokenized.)
    if value.is_ascii() && tokens.iter().all(|t| t.is_ascii()) {
        lowered.clear();
        lowered.push_str(value);
        lowered.make_ascii_lowercase();
        let present = |t: &String| lowered.contains(t.as_str());
        let possible = match mode {
            TextMatchMode::Any => tokens.iter().any(present),
            TextMatchMode::All => tokens.iter().all(present),
        };
        if !possible {
            return false;
        }
    }
    let mut found = vec![false; tokens.len()];
    let mut left = tokens.len();
    let mut done = false;
    tokenizer.tokenize_each_inline(value, |token| {
        if done {
            return;
        }
        if let Some(i) = tokens.iter().position(|t| t == token) {
            if !found[i] {
                found[i] = true;
                left -= 1;
            }
            done = mode == TextMatchMode::Any || left == 0;
        }
    });
    match mode {
        TextMatchMode::Any => left < tokens.len(),
        TextMatchMode::All => left == 0,
    }
}

fn match_strings<'a>(
    values: impl Iterator<Item = Option<&'a str>>,
    len: usize,
    tokens: &[String],
    mode: TextMatchMode,
) -> BooleanArray {
    let tokenizer = StandardTokenizer::new();
    let mut out = BooleanBuilder::with_capacity(len);
    let mut lowered = String::new();
    for value in values {
        match value {
            Some(value) => out.append_value(holds(&tokenizer, value, tokens, mode, &mut lowered)),
            None => out.append_null(),
        }
    }
    out.finish()
}

impl ScalarUDFImpl for TextMatchFunc {
    fn name(&self) -> &str {
        TEXT_MATCH_UDF
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DfResult<DataType> {
        if !(2..=3).contains(&arg_types.len()) {
            return plan_err!("{TEXT_MATCH_UDF} takes (column, query [, mode])");
        }
        if let Some(bad) = arg_types.iter().find(|t| !is_string(t)) {
            return plan_err!("{TEXT_MATCH_UDF} takes string arguments, not {bad}");
        }
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let query = literal_arg(&args[1], "query")?;
        let mode = match args.get(2) {
            Some(mode) => match literal_arg(mode, "mode")? {
                Some(mode) => match TextMatchMode::parse(mode) {
                    Some(mode) => mode,
                    None => {
                        return exec_err!(
                            "{TEXT_MATCH_UDF}: mode must be 'and' or 'or', not {mode:?}"
                        );
                    }
                },
                None => return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None))),
            },
            None => TextMatchMode::Any,
        };
        let Some(query) = query else {
            return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)));
        };
        let tokens = query_tokens(query);
        let column: ArrayRef = match &args[0] {
            ColumnarValue::Array(array) => Arc::clone(array),
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(number_rows.max(1))?,
        };
        let result = match column.data_type() {
            DataType::Utf8 => {
                let a = column.as_any().downcast_ref::<StringArray>().expect("utf8");
                match_strings(a.iter(), a.len(), &tokens, mode)
            }
            DataType::LargeUtf8 => {
                let a = column
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("large utf8");
                match_strings(a.iter(), a.len(), &tokens, mode)
            }
            DataType::Utf8View => {
                let a = column
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .expect("utf8 view");
                match_strings(a.iter(), a.len(), &tokens, mode)
            }
            DataType::Null => BooleanArray::new_null(column.len()),
            other => {
                return exec_err!("{TEXT_MATCH_UDF}: the column must be a string, not {other}");
            }
        };
        match &args[0] {
            ColumnarValue::Scalar(_) => Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?)),
            ColumnarValue::Array(_) => Ok(ColumnarValue::Array(Arc::new(result))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(values: &[Option<&str>], query: &str, mode: TextMatchMode) -> Vec<Option<bool>> {
        let tokens = query_tokens(query);
        match_strings(values.iter().copied(), values.len(), &tokens, mode)
            .iter()
            .collect()
    }

    #[test]
    fn a_value_matches_on_whole_tokens_any_or_all() {
        let values = [
            Some("A monkey and a bear"),
            Some("monkeys only"),
            Some("Jackal!"),
            None,
            Some(""),
        ];
        assert_eq!(
            eval(&values, "monkey jackal", TextMatchMode::Any),
            vec![Some(true), Some(false), Some(true), None, Some(false)]
        );
        assert_eq!(
            eval(&values, "bear MONKEY", TextMatchMode::All),
            vec![Some(true), Some(false), Some(false), None, Some(false)]
        );
    }

    #[test]
    fn the_substring_check_never_drops_a_token() {
        // Mixed case, a token inside punctuation, a cut-length word, and
        // non-ASCII text that lowercases into an ASCII token.
        let long = "x".repeat(300);
        let values = [
            Some("The BBC's REPORT"),
            Some("bbc-funded (BBC)"),
            Some(long.as_str()),
            Some("\u{212A}elvin"),
            Some("Kelvinator"),
        ];
        assert_eq!(
            eval(&values, "bbc's kelvin", TextMatchMode::Any),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
        assert_eq!(
            eval(&values, &long, TextMatchMode::All),
            vec![
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn a_query_without_tokens_matches_nothing() {
        assert_eq!(
            eval(&[Some("anything")], " ,; ", TextMatchMode::Any),
            vec![Some(false)]
        );
        assert_eq!(
            eval(&[Some("anything")], "", TextMatchMode::All),
            vec![Some(false)]
        );
    }

    #[test]
    fn query_tokens_are_the_standard_tokens_once_each() {
        assert_eq!(
            query_tokens("Rust rust, \"async\" +runtime"),
            vec!["rust", "async", "runtime"]
        );
    }
}
