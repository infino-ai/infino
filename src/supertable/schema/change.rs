// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How the schema changes: the one declarative write, merged into the
//! current document, and the changes that merge produces.
//!
//! A caller submits a [`SchemaPatch`] — the shape [`TableSchema`] reads
//! back as, so applying what was read commits nothing — and [`merge`]
//! turns the difference into [`SchemaChange`]s, which
//! [`TableSchema::apply`] folds into the next document. A column is
//! matched by id when the patch carries one and by name otherwise; a
//! column the patch does not mention is untouched. Nothing is ever
//! dropped by omission.

use arrow_schema::DataType;
use serde_json::{Map, Value};

use super::{
    ColumnIndex, FieldDef, FieldId, MAX_MAX_DEPTH, MAX_MAX_FIELDS, MAX_TEMPLATE_WILDCARDS,
    TableSchema, Template,
    error::SchemaError,
    template_to_json, templates_from_json,
    types::{data_type_from_keys, type_keys},
};

/// One change to the schema. Not public API: the write is the patch.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaChange {
    /// A new column, minted the next id. The same change an append emits
    /// for a column it does not know.
    AddColumn {
        name: String,
        data_type: DataType,
        nullable: bool,
        index: Option<ColumnIndex>,
    },
    /// Retire a column: its id is tombstoned and its index config goes.
    DropColumn { id: FieldId },
    /// A label change only.
    RenameColumn { id: FieldId, to: String },
    /// A lossless type change: metadata only, files cast on read.
    WidenColumn { id: FieldId, to: DataType },
    /// A lossy type change: the type flips at once and files are
    /// converted by compaction; `converting_from` records the old type
    /// until the last file is rewritten. A value that does not cast
    /// becomes null, so the column admits nulls from the flip on.
    RewriteColumn { id: FieldId, to: DataType },
    /// A nullable column becomes non-nullable (empty tables only).
    SetNonNullable { id: FieldId },
    /// A non-nullable column becomes nullable.
    SetNullable { id: FieldId },
    /// The field cap.
    SetMaxFields(u32),
    /// The nesting cap for documents.
    SetMaxDepth(u32),
    /// The rules for columns documents add, replacing the current ones.
    SetTemplates(Vec<Template>),
}

/// How a type change is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeChange {
    /// Every value of the old type is exactly representable in the new.
    Widen,
    /// Values may not survive; what does not cast becomes null.
    Rewrite,
}

/// Whether `from` → `to` loses nothing.
pub fn classify_type_change(from: &DataType, to: &DataType) -> TypeChange {
    use DataType::*;
    fn rank_int(dt: &DataType) -> Option<u8> {
        Some(match dt {
            Int8 => 1,
            Int16 => 2,
            Int32 => 3,
            Int64 => 4,
            _ => return None,
        })
    }
    fn rank_uint(dt: &DataType) -> Option<u8> {
        Some(match dt {
            UInt8 => 1,
            UInt16 => 2,
            UInt32 => 3,
            UInt64 => 4,
            _ => return None,
        })
    }
    let widens = match (from, to) {
        (Null, _) => true,
        (Float32, Float64) | (Utf8, LargeUtf8) | (Binary, LargeBinary) => true,
        (Date32, Timestamp(_, None)) => true,
        (Decimal128(p, s), Decimal128(p2, s2)) => p2 > p && s == s2,
        (List(a), List(b)) | (LargeList(a), LargeList(b)) => {
            a.data_type() == b.data_type()
                || classify_type_change(a.data_type(), b.data_type()) == TypeChange::Widen
        }
        _ => match (rank_int(from), rank_int(to), rank_uint(from), rank_uint(to)) {
            (Some(a), Some(b), _, _) => a < b,
            (_, _, Some(a), Some(b)) => a < b,
            _ => false,
        },
    };
    if widens {
        TypeChange::Widen
    } else {
        TypeChange::Rewrite
    }
}

/// One field of a [`SchemaPatch`].
///
/// Build it with [`FieldPatch::named`] and the `with_*` setters rather than
/// a struct literal: the type is `#[non_exhaustive]` so it can grow a field
/// without breaking callers.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct FieldPatch {
    /// Set to address a live column by identity (the one way to rename).
    pub id: Option<FieldId>,
    /// The column's name: the one it has, or the one it gets.
    pub name: String,
    /// Required for a new column; a different type on a live column
    /// changes it.
    pub data_type: Option<DataType>,
    /// Whether the column admits nulls; a new column does by default.
    pub nullable: Option<bool>,
    /// An index for a new column. On a live column it must match the
    /// column's: an analyzer chain or a metric is part of the identity.
    pub index: Option<ColumnIndex>,
    /// Drop the column.
    pub dropped: bool,
}

/// What a caller writes: fields to add or change, and the cap.
///
/// Build it with [`SchemaPatch::new`] and the `with_*` setters rather than a
/// struct literal: the type is `#[non_exhaustive]` so it can grow a field
/// without breaking callers.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct SchemaPatch {
    /// The columns to add or change. A live column not listed is untouched.
    pub fields: Vec<FieldPatch>,
    /// The field cap to set, when present.
    pub max_fields: Option<u32>,
    /// The document nesting cap to set, when present.
    pub max_depth: Option<u32>,
    /// The templates to set, replacing the current ones, when present.
    pub templates: Option<Vec<Template>>,
}

impl FieldPatch {
    /// A patch for the column called `name`: the one to add, or the live one
    /// to change. Add [`Self::with_id`] to address a live column by identity
    /// instead, which is the one way to rename it.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Address the live column with this id, so `name` renames it.
    pub fn with_id(mut self, id: FieldId) -> Self {
        self.id = Some(id);
        self
    }

    /// The column's type: required for a new column, a change for a live one.
    pub fn with_type(mut self, data_type: DataType) -> Self {
        self.data_type = Some(data_type);
        self
    }

    /// Whether the column admits nulls.
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = Some(nullable);
        self
    }

    /// The index a new column carries.
    pub fn with_index(mut self, index: ColumnIndex) -> Self {
        self.index = Some(index);
        self
    }

    /// Retire the column.
    pub fn dropped(mut self) -> Self {
        self.dropped = true;
        self
    }
}

impl SchemaPatch {
    /// A patch over `fields`, changing no cap.
    pub fn new(fields: Vec<FieldPatch>) -> Self {
        Self {
            fields,
            ..Self::default()
        }
    }

    /// Set the field cap.
    pub fn with_max_fields(mut self, max_fields: u32) -> Self {
        self.max_fields = Some(max_fields);
        self
    }

    /// Set the document nesting cap.
    pub fn with_max_depth(mut self, max_depth: u32) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Replace the templates that decide what a new column becomes.
    pub fn with_templates(mut self, templates: Vec<Template>) -> Self {
        self.templates = Some(templates);
        self
    }
}

impl From<&TableSchema> for SchemaPatch {
    /// The document as a patch: applying it changes nothing.
    fn from(schema: &TableSchema) -> Self {
        Self {
            fields: schema
                .fields()
                .iter()
                .map(|f| FieldPatch {
                    id: Some(f.id),
                    name: f.name.clone(),
                    data_type: Some(f.data_type.clone()),
                    nullable: Some(f.nullable),
                    index: f.index.clone(),
                    dropped: false,
                })
                .collect(),
            max_fields: Some(schema.max_fields()),
            max_depth: Some(schema.max_depth()),
            templates: Some(schema.templates().to_vec()),
        }
    }
}

impl SchemaPatch {
    /// The patch `json` spells: the document's keys (`fields` with `id`,
    /// `name`, type keys, `nullable`, `index`, `dropped`; `max_fields`),
    /// any other key ignored so a read document applies as is.
    pub fn from_json(json: &Value) -> Result<Self, String> {
        let doc = json
            .as_object()
            .ok_or_else(|| "schema patch is not an object".to_string())?;
        let fields = doc
            .get("fields")
            .and_then(Value::as_array)
            .map(|fields| {
                fields
                    .iter()
                    .map(field_patch_from_json)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        let cap = |key: &str| -> Result<Option<u32>, String> {
            doc.get(key)
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| format!("{key} is not a u32"))
                })
                .transpose()
        };
        let templates = doc
            .get("templates")
            .map(|t| templates_from_json(Some(t)))
            .transpose()?;
        Ok(Self {
            fields,
            max_fields: cap("max_fields")?,
            max_depth: cap("max_depth")?,
            templates,
        })
    }

    /// The patch as JSON, in the document's spelling plus `dropped`.
    pub fn to_json(&self) -> Value {
        let fields: Vec<Value> = self
            .fields
            .iter()
            .map(|f| {
                let mut keys = match &f.data_type {
                    Some(dt) => type_keys(dt),
                    None => Map::new(),
                };
                if let Some(id) = f.id {
                    keys.insert("id".into(), Value::from(id.0));
                }
                keys.insert("name".into(), Value::from(f.name.as_str()));
                if let Some(nullable) = f.nullable {
                    keys.insert("nullable".into(), Value::from(nullable));
                }
                if let Some(index) = &f.index {
                    keys.insert("index".into(), super::index_to_json(index));
                }
                if f.dropped {
                    keys.insert("dropped".into(), Value::from(true));
                }
                Value::Object(keys)
            })
            .collect();
        let mut doc = Map::new();
        doc.insert("fields".into(), Value::Array(fields));
        if let Some(cap) = self.max_fields {
            doc.insert("max_fields".into(), Value::from(cap));
        }
        if let Some(cap) = self.max_depth {
            doc.insert("max_depth".into(), Value::from(cap));
        }
        if let Some(templates) = &self.templates {
            doc.insert(
                "templates".into(),
                Value::Array(templates.iter().map(template_to_json).collect()),
            );
        }
        Value::Object(doc)
    }
}

fn field_patch_from_json(json: &Value) -> Result<FieldPatch, String> {
    let keys = json
        .as_object()
        .ok_or_else(|| "field is not an object".to_string())?;
    let id = keys
        .get("id")
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .map(FieldId)
                .ok_or_else(|| "field id is not a u32".to_string())
        })
        .transpose()?;
    let name = keys
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "field has no name".to_string())?
        .to_owned();
    let data_type = keys
        .contains_key("type")
        .then(|| data_type_from_keys(keys))
        .transpose()?;
    let nullable = keys.get("nullable").and_then(Value::as_bool);
    let index = keys.get("index").map(super::index_from_json).transpose()?;
    let dropped = keys
        .get("dropped")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(FieldPatch {
        id,
        name,
        data_type,
        nullable,
        index,
        dropped,
    })
}

/// What the merge needs to know about the table beyond its schema.
#[derive(Debug, Clone, Copy)]
pub struct MergeContext<'a> {
    /// The id column's name; a patch cannot name it.
    pub id_column: &'a str,
    /// Whether the table holds no rows (nullability can then tighten).
    pub table_empty: bool,
    /// The column the global vector index is built on, if any.
    pub vector_index_column: Option<&'a str>,
}

/// The changes `patch` makes to `current`, or why it is refused.
pub fn merge(
    current: &TableSchema,
    patch: &SchemaPatch,
    ctx: MergeContext<'_>,
) -> Result<Vec<SchemaChange>, SchemaError> {
    let mut changes = Vec::new();
    // Names live columns will have once the patch applies, for collision
    // checks: start from the current names and follow renames and drops.
    let mut names: Vec<(FieldId, String)> = current
        .fields()
        .iter()
        .map(|f| (f.id, f.name.clone()))
        .collect();
    let mut new_names: Vec<String> = Vec::new();

    for field in &patch.fields {
        if field.name == ctx.id_column {
            return Err(SchemaError::NameTaken {
                name: field.name.clone(),
            });
        }
        let live: Option<&FieldDef> = match field.id {
            Some(id) => Some(
                current
                    .fields()
                    .iter()
                    .find(|f| f.id == id)
                    .ok_or(SchemaError::UnknownFieldId { id })?,
            ),
            None => current.fields().iter().find(|f| f.name == field.name),
        };
        let Some(live) = live else {
            if field.dropped {
                // Dropping a column that is not there is nothing to do.
                continue;
            }
            let data_type = field
                .data_type
                .clone()
                .ok_or_else(|| SchemaError::TypeRequired {
                    column: field.name.clone(),
                })?;
            let nullable = field.nullable.unwrap_or(true);
            if !nullable && !ctx.table_empty {
                return Err(SchemaError::NotEmpty {
                    column: field.name.clone(),
                });
            }
            if names.iter().any(|(_, n)| *n == field.name) || new_names.contains(&field.name) {
                return Err(SchemaError::NameTaken {
                    name: field.name.clone(),
                });
            }
            new_names.push(field.name.clone());
            changes.push(SchemaChange::AddColumn {
                name: field.name.clone(),
                data_type,
                nullable,
                index: field.index.clone(),
            });
            continue;
        };

        if field.dropped {
            if ctx.vector_index_column == Some(live.name.as_str()) {
                return Err(SchemaError::BacksGlobalVectorIndex {
                    column: live.name.clone(),
                });
            }
            names.retain(|(id, _)| *id != live.id);
            changes.push(SchemaChange::DropColumn { id: live.id });
            continue;
        }
        if field.name != live.name {
            if names
                .iter()
                .any(|(id, n)| *id != live.id && *n == field.name)
                || new_names.contains(&field.name)
            {
                return Err(SchemaError::NameTaken {
                    name: field.name.clone(),
                });
            }
            if let Some(entry) = names.iter_mut().find(|(id, _)| *id == live.id) {
                entry.1 = field.name.clone();
            }
            changes.push(SchemaChange::RenameColumn {
                id: live.id,
                to: field.name.clone(),
            });
        }
        if let Some(index) = &field.index
            && Some(index) != live.index.as_ref()
        {
            return Err(SchemaError::IdentityChange {
                attribute: format!("index of column `{}`", live.name),
            });
        }
        if let Some(to) = &field.data_type
            && *to != live.data_type
        {
            // While files in the old type remain, the one type change
            // allowed is back to it: those files then read their originals
            // through the identity path, and the files converted in between
            // are cast back.
            if live.converting_from.is_some() && live.converting_from.as_ref() != Some(to) {
                return Err(SchemaError::ConversionInProgress {
                    column: live.name.clone(),
                });
            }
            if ctx.vector_index_column == Some(live.name.as_str()) {
                return Err(SchemaError::BacksGlobalVectorIndex {
                    column: live.name.clone(),
                });
            }
            changes.push(match classify_type_change(&live.data_type, to) {
                TypeChange::Widen => SchemaChange::WidenColumn {
                    id: live.id,
                    to: to.clone(),
                },
                TypeChange::Rewrite => SchemaChange::RewriteColumn {
                    id: live.id,
                    to: to.clone(),
                },
            });
        }
        match field.nullable {
            Some(false) if live.nullable => {
                if !ctx.table_empty {
                    return Err(SchemaError::NotEmpty {
                        column: live.name.clone(),
                    });
                }
                changes.push(SchemaChange::SetNonNullable { id: live.id });
            }
            Some(true) if !live.nullable => changes.push(SchemaChange::SetNullable { id: live.id }),
            _ => {}
        }
    }
    if let Some(cap) = patch.max_fields
        && cap != current.max_fields()
    {
        changes.push(SchemaChange::SetMaxFields(cap));
    }
    if let Some(cap) = patch.max_depth
        && cap != current.max_depth()
    {
        changes.push(SchemaChange::SetMaxDepth(cap));
    }
    if let Some(templates) = &patch.templates
        && templates.as_slice() != current.templates()
    {
        changes.push(SchemaChange::SetTemplates(templates.clone()));
    }
    Ok(changes)
}

impl TableSchema {
    /// The document after `changes`, with `schema_id` advanced once when
    /// anything changed. Ids are minted here and nowhere else.
    pub(crate) fn apply(&self, changes: &[SchemaChange]) -> Result<TableSchema, SchemaError> {
        if changes.is_empty() {
            return Ok(self.clone());
        }
        let mut next = self.clone();
        for change in changes {
            next.apply_one(change)?;
        }
        next.schema_id += 1;
        Ok(next)
    }

    fn apply_one(&mut self, change: &SchemaChange) -> Result<(), SchemaError> {
        match change {
            SchemaChange::AddColumn {
                name,
                data_type,
                nullable,
                index,
            } => {
                if self.fields.iter().any(|f| &f.name == name) {
                    return Err(SchemaError::NameTaken { name: name.clone() });
                }
                if self.fields.len() as u32 + 1 > self.max_fields {
                    return Err(SchemaError::FieldCapExceeded {
                        cap: self.max_fields,
                        current: self.fields.len() as u32,
                        fields: vec![name.clone()],
                    });
                }
                check_type_buildable(name, data_type)?;
                if let Some(index) = index {
                    check_index_fits(name, index, data_type)?;
                }
                self.last_field_id += 1;
                self.fields.push(FieldDef {
                    id: FieldId(self.last_field_id),
                    name: name.clone(),
                    data_type: data_type.clone(),
                    nullable: *nullable,
                    index: index.clone(),
                    converting_from: None,
                });
            }
            SchemaChange::DropColumn { id } => {
                let before = self.fields.len();
                self.fields.retain(|f| f.id != *id);
                if self.fields.len() == before {
                    return Err(SchemaError::UnknownFieldId { id: *id });
                }
                self.tombstoned.push(*id);
            }
            SchemaChange::RenameColumn { id, to } => {
                if self.fields.iter().any(|f| f.id != *id && f.name == *to) {
                    return Err(SchemaError::NameTaken { name: to.clone() });
                }
                self.field_mut(*id)?.name = to.clone();
            }
            SchemaChange::WidenColumn { id, to } => {
                let name = self.field_mut(*id)?.name.clone();
                check_type_buildable(&name, to)?;
                let field = self.field_mut(*id)?;
                field.data_type = to.clone();
                field.index = index_after_retype(field.index.take(), to);
            }
            SchemaChange::RewriteColumn { id, to } => {
                let name = self.field_mut(*id)?.name.clone();
                check_type_buildable(&name, to)?;
                let field = self.field_mut(*id)?;
                if field.converting_from.is_some() && field.converting_from.as_ref() != Some(to) {
                    return Err(SchemaError::ConversionInProgress {
                        column: field.name.clone(),
                    });
                }
                field.converting_from = Some(field.data_type.clone());
                field.data_type = to.clone();
                field.nullable = true;
                field.index = index_after_retype(field.index.take(), to);
            }
            SchemaChange::SetNonNullable { id } => self.field_mut(*id)?.nullable = false,
            SchemaChange::SetNullable { id } => self.field_mut(*id)?.nullable = true,
            SchemaChange::SetMaxFields(cap) => {
                self.max_fields = check_cap("max_fields", *cap, MAX_MAX_FIELDS)?;
            }
            SchemaChange::SetMaxDepth(cap) => {
                self.max_depth = check_cap("max_depth", *cap, MAX_MAX_DEPTH)?;
            }
            SchemaChange::SetTemplates(templates) => {
                for template in templates {
                    let wildcards = template.path.matches('*').count();
                    if wildcards > MAX_TEMPLATE_WILDCARDS {
                        return Err(SchemaError::CapExceeded {
                            setting: format!("wildcards in template `{}`", template.name),
                            value: wildcards as u32,
                            cap: MAX_TEMPLATE_WILDCARDS as u32,
                        });
                    }
                }
                self.templates = templates.clone();
            }
        }
        Ok(())
    }

    fn field_mut(&mut self, id: FieldId) -> Result<&mut FieldDef, SchemaError> {
        self.fields
            .iter_mut()
            .find(|f| f.id == id)
            .ok_or(SchemaError::UnknownFieldId { id })
    }

    /// The document with the conversions of the columns in `ids` complete:
    /// their old types are forgotten. `None` when none of them was
    /// converting.
    pub(crate) fn with_conversions_cleared(&self, ids: &[FieldId]) -> Option<TableSchema> {
        let mut next = self.clone();
        let mut cleared = false;
        for f in next.fields.iter_mut() {
            if ids.contains(&f.id) && f.converting_from.take().is_some() {
                cleared = true;
            }
        }
        if !cleared {
            return None;
        }
        next.schema_id += 1;
        Some(next)
    }

    /// The columns whose conversion is outstanding.
    pub(crate) fn converting(&self) -> impl Iterator<Item = FieldId> + '_ {
        self.fields
            .iter()
            .filter(|f| f.converting_from.is_some())
            .map(|f| f.id)
    }
}

/// `value` if it is within `cap`, else the refusal naming both. A cap the
/// owner sets is itself capped: these bound what one commit can cost, so
/// raising one past what the engine can carry would remove the bound
/// rather than widen it.
fn check_cap(setting: &str, value: u32, cap: u32) -> Result<u32, SchemaError> {
    if value > cap {
        return Err(SchemaError::CapExceeded {
            setting: setting.to_owned(),
            value,
            cap,
        });
    }
    Ok(value)
}

/// Whether `index` can be built on a new column of type `data_type`: a
/// full-text index needs a string column, and a vector index is declared
/// when the table is created, because the index's storage is laid out
/// with the table.
/// Refuse a type the engine cannot build an array from. Arrow sizes a
/// fixed-size list's child buffer as `size * rows`, so a negative size
/// overflows that multiplication the first time a column is null-filled,
/// which aborts the writer rather than failing the call. The check is
/// recursive: a negative size is just as fatal nested inside a list or a
/// struct as it is at the top.
fn check_type_buildable(column: &str, data_type: &DataType) -> Result<(), SchemaError> {
    let invalid = |reason: String| {
        Err(SchemaError::InvalidType {
            column: column.to_owned(),
            reason,
        })
    };
    match data_type {
        DataType::FixedSizeList(item, size) => {
            if *size <= 0 {
                return invalid(format!(
                    "a fixed-size list needs a positive size, not {size}"
                ));
            }
            check_type_buildable(column, item.data_type())
        }
        DataType::FixedSizeBinary(width) => {
            if *width <= 0 {
                return invalid(format!(
                    "fixed-size binary needs a positive width, not {width}"
                ));
            }
            Ok(())
        }
        DataType::List(item) | DataType::LargeList(item) => {
            check_type_buildable(column, item.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .try_for_each(|f| check_type_buildable(column, f.data_type())),
        _ => Ok(()),
    }
}

fn check_index_fits(
    column: &str,
    index: &ColumnIndex,
    data_type: &DataType,
) -> Result<(), SchemaError> {
    match index {
        ColumnIndex::Fts { .. }
            if !matches!(
                data_type,
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            ) =>
        {
            Err(SchemaError::InvalidIndex {
                column: column.to_owned(),
                reason: format!("a full-text index needs a string column, not `{data_type}`"),
            })
        }
        ColumnIndex::Fts { .. } => Ok(()),
        ColumnIndex::Vector { .. } => Err(SchemaError::InvalidIndex {
            column: column.to_owned(),
            reason: "a vector index is declared when the table is created".to_owned(),
        }),
    }
}

/// The index a column keeps after its type changes to `to`: a full-text
/// index survives a string-to-string change, nothing else does.
fn index_after_retype(index: Option<ColumnIndex>, to: &DataType) -> Option<ColumnIndex> {
    match index {
        Some(ColumnIndex::Fts { .. })
            if matches!(
                to,
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            ) =>
        {
            index
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Schema, TimeUnit};

    use super::*;
    use crate::superfile::builder::FtsConfig;

    fn table() -> TableSchema {
        TableSchema::from_options(
            &Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, false),
                Field::new("score", DataType::Int64, true),
            ]),
            &[FtsConfig::new("title")],
            &[],
        )
    }

    fn ctx(empty: bool) -> MergeContext<'static> {
        MergeContext {
            id_column: "_id",
            table_empty: empty,
            vector_index_column: None,
        }
    }

    fn add(name: &str, dt: DataType) -> FieldPatch {
        FieldPatch {
            id: None,
            name: name.into(),
            data_type: Some(dt),
            nullable: None,
            index: None,
            dropped: false,
        }
    }

    #[test]
    fn the_read_document_applies_as_a_no_op_and_so_does_any_subset() {
        let t = table();
        let full = SchemaPatch::from(&t);
        assert!(merge(&t, &full, ctx(false)).expect("merge").is_empty());
        let round = SchemaPatch::from_json(&t.to_json()).expect("the document is a patch");
        assert_eq!(round.max_depth, Some(t.max_depth()));
        assert_eq!(round.templates.as_deref(), Some(t.templates()));
        assert!(merge(&t, &round, ctx(false)).expect("merge").is_empty());
        for field in &full.fields {
            let one = SchemaPatch {
                fields: vec![field.clone()],
                max_fields: None,
                max_depth: None,
                templates: None,
            };
            assert!(
                merge(&t, &one, ctx(false)).expect("merge").is_empty(),
                "{}",
                field.name
            );
        }
        assert_eq!(t.apply(&[]).expect("apply").schema_id(), t.schema_id());
    }

    #[test]
    fn adds_renames_drops_and_caps_merge_and_apply() {
        let t = table();
        let patch = SchemaPatch {
            fields: vec![
                add("tag", DataType::LargeUtf8),
                FieldPatch {
                    id: Some(FieldId(2)),
                    name: "points".into(),
                    data_type: None,
                    nullable: None,
                    index: None,
                    dropped: false,
                },
            ],
            max_fields: Some(50),
            max_depth: None,
            templates: None,
        };
        let changes = merge(&t, &patch, ctx(false)).expect("merge");
        assert_eq!(
            changes,
            vec![
                SchemaChange::AddColumn {
                    name: "tag".into(),
                    data_type: DataType::LargeUtf8,
                    nullable: true,
                    index: None
                },
                SchemaChange::RenameColumn {
                    id: FieldId(2),
                    to: "points".into()
                },
                SchemaChange::SetMaxFields(50),
            ]
        );
        let next = t.apply(&changes).expect("apply");
        assert_eq!(next.schema_id(), 2);
        assert_eq!(next.last_field_id(), 3);
        assert_eq!(next.id_of("tag"), Some(FieldId(3)));
        assert_eq!(next.name_of(FieldId(2)), Some("points"));
        assert_eq!(next.max_fields(), 50);

        let drop = SchemaPatch {
            fields: vec![FieldPatch {
                dropped: true,
                ..add("points", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        let dropped = next
            .apply(&merge(&next, &drop, ctx(false)).expect("merge"))
            .expect("apply");
        assert_eq!(dropped.id_of("points"), None);
        assert_eq!(dropped.tombstoned(), &[FieldId(2)]);
        // Adding the dropped name again mints a new id; renaming onto a
        // dropped name is fine.
        let again = dropped
            .apply(
                &merge(
                    &dropped,
                    &SchemaPatch {
                        fields: vec![add("points", DataType::Int64)],
                        max_fields: None,
                        max_depth: None,
                        templates: None,
                    },
                    ctx(false),
                )
                .expect("merge"),
            )
            .expect("apply");
        assert_eq!(again.id_of("points"), Some(FieldId(4)));
    }

    #[test]
    fn the_refusals_name_their_rule() {
        let t = table();
        let unknown = SchemaPatch {
            fields: vec![FieldPatch {
                id: Some(FieldId(9)),
                ..add("x", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(matches!(
            merge(&t, &unknown, ctx(false)),
            Err(SchemaError::UnknownFieldId { id: FieldId(9) })
        ));
        let taken = SchemaPatch {
            fields: vec![FieldPatch {
                id: Some(FieldId(2)),
                name: "title".into(),
                ..add("", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(
            matches!(merge(&t, &taken, ctx(false)), Err(SchemaError::NameTaken { name }) if name == "title")
        );
        let id_col = SchemaPatch {
            fields: vec![add("_id", DataType::Int64)],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(matches!(
            merge(&t, &id_col, ctx(false)),
            Err(SchemaError::NameTaken { .. })
        ));
        let analyzer = SchemaPatch {
            fields: vec![FieldPatch {
                index: Some(ColumnIndex::Fts {
                    analyzer: "ascii_lower".into(),
                    stopwords: Default::default(),
                    stemmer: Default::default(),
                    positions: false,
                    stored: true,
                    bm25: crate::superfile::fts::bm25::Bm25Params::STANDARD,
                }),
                ..add("title", DataType::LargeUtf8)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(matches!(
            merge(&t, &analyzer, ctx(false)),
            Err(SchemaError::IdentityChange { .. })
        ));
        let tighten = SchemaPatch {
            fields: vec![FieldPatch {
                nullable: Some(false),
                ..add("score", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(
            matches!(merge(&t, &tighten, ctx(false)), Err(SchemaError::NotEmpty { column }) if column == "score")
        );
        assert_eq!(
            merge(&t, &tighten, ctx(true)).expect("empty table"),
            vec![SchemaChange::SetNonNullable { id: FieldId(2) }]
        );
        let non_null_add = SchemaPatch {
            fields: vec![FieldPatch {
                nullable: Some(false),
                ..add("tag", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(matches!(
            merge(&t, &non_null_add, ctx(false)),
            Err(SchemaError::NotEmpty { .. })
        ));
        let gvi = MergeContext {
            id_column: "_id",
            table_empty: false,
            vector_index_column: Some("score"),
        };
        let drop_gvi = SchemaPatch {
            fields: vec![FieldPatch {
                dropped: true,
                ..add("score", DataType::Int64)
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        assert!(matches!(
            merge(&t, &drop_gvi, gvi),
            Err(SchemaError::BacksGlobalVectorIndex { .. })
        ));
    }

    #[test]
    fn type_changes_classify_and_flip_with_the_index_following() {
        for (from, to) in [
            (DataType::Int8, DataType::Int64),
            (DataType::UInt8, DataType::UInt64),
            (DataType::Float32, DataType::Float64),
            (DataType::Decimal128(10, 2), DataType::Decimal128(20, 2)),
            (DataType::Utf8, DataType::LargeUtf8),
            (DataType::Binary, DataType::LargeBinary),
            (
                DataType::Date32,
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            (
                DataType::List(std::sync::Arc::new(Field::new(
                    "item",
                    DataType::Int32,
                    true,
                ))),
                DataType::List(std::sync::Arc::new(Field::new(
                    "item",
                    DataType::Int64,
                    true,
                ))),
            ),
            (DataType::Null, DataType::Boolean),
        ] {
            assert_eq!(
                classify_type_change(&from, &to),
                TypeChange::Widen,
                "{from} -> {to}"
            );
        }
        for (from, to) in [
            (DataType::Int64, DataType::Int32),
            (DataType::Int64, DataType::Float64),
            (DataType::LargeUtf8, DataType::Int64),
            (DataType::Boolean, DataType::Int64),
            (DataType::Decimal128(20, 2), DataType::Decimal128(20, 3)),
        ] {
            assert_eq!(
                classify_type_change(&from, &to),
                TypeChange::Rewrite,
                "{from} -> {to}"
            );
        }

        let t = table();
        let retype = SchemaPatch {
            fields: vec![add("title", DataType::Int64)],
            max_fields: None,
            max_depth: None,
            templates: None,
        };
        let changes = merge(&t, &retype, ctx(false)).expect("merge");
        assert_eq!(
            changes,
            vec![SchemaChange::RewriteColumn {
                id: FieldId(1),
                to: DataType::Int64
            }]
        );
        let flipped = t.apply(&changes).expect("apply");
        let title = &flipped.fields()[0];
        assert_eq!(title.data_type, DataType::Int64);
        assert_eq!(title.converting_from, Some(DataType::LargeUtf8));
        assert!(title.nullable, "what does not cast becomes null");
        assert!(
            title.index.is_none(),
            "a full-text index does not survive a retype to integers"
        );
        assert!(matches!(
            merge(
                &flipped,
                &SchemaPatch {
                    fields: vec![add("title", DataType::Float64)],
                    max_fields: None,
                    max_depth: None,
                    templates: None
                },
                ctx(false)
            ),
            Err(SchemaError::ConversionInProgress { .. })
        ));
        assert_eq!(flipped.converting().collect::<Vec<_>>(), vec![FieldId(1)]);
        // The one change allowed while converting: back to the old type.
        let back = merge(
            &flipped,
            &SchemaPatch {
                fields: vec![add("title", DataType::LargeUtf8)],
                max_fields: None,
                max_depth: None,
                templates: None,
            },
            ctx(false),
        )
        .expect("flip back");
        assert_eq!(
            back,
            vec![SchemaChange::RewriteColumn {
                id: FieldId(1),
                to: DataType::LargeUtf8
            }]
        );
        let returned = flipped.apply(&back).expect("apply");
        assert_eq!(returned.fields()[0].data_type, DataType::LargeUtf8);
        assert_eq!(returned.fields()[0].converting_from, Some(DataType::Int64));
        let cleared = flipped
            .with_conversions_cleared(&[FieldId(1)])
            .expect("was converting");
        assert!(cleared.fields()[0].converting_from.is_none());
        assert_eq!(cleared.schema_id(), flipped.schema_id() + 1);
        assert!(cleared.with_conversions_cleared(&[FieldId(1)]).is_none());
        assert_eq!(cleared.converting().count(), 0);

        let widen = merge(
            &t,
            &SchemaPatch {
                fields: vec![add("score", DataType::Decimal128(38, 0))],
                max_fields: None,
                max_depth: None,
                templates: None,
            },
            ctx(false),
        )
        .expect("merge");
        assert_eq!(
            widen,
            vec![SchemaChange::RewriteColumn {
                id: FieldId(2),
                to: DataType::Decimal128(38, 0)
            }]
        );
        let widened = t
            .apply(&[SchemaChange::WidenColumn {
                id: FieldId(2),
                to: DataType::Float64,
            }])
            .expect("apply");
        assert_eq!(widened.fields()[1].data_type, DataType::Float64);
        assert!(widened.fields()[1].converting_from.is_none());
    }

    #[test]
    fn a_patch_round_trips_through_json_with_dropped() {
        let patch = SchemaPatch {
            fields: vec![
                FieldPatch {
                    dropped: true,
                    ..add("old", DataType::Int64)
                },
                FieldPatch {
                    id: Some(FieldId(2)),
                    ..add("points", DataType::Int64)
                },
                FieldPatch {
                    data_type: None,
                    ..add("renamed_only", DataType::Null)
                },
            ],
            max_fields: Some(7),
            max_depth: None,
            templates: None,
        };
        let back = SchemaPatch::from_json(&patch.to_json()).expect("decode");
        assert_eq!(back, patch);
    }
}
