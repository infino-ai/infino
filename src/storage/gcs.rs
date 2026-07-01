// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Google Cloud Storage-backed [`StorageProvider`].
//!
//! Wraps `object_store::gcp::GoogleCloudStorage` so the supertable
//! exercises the same code paths on GCS as on S3, Azure, and LocalFS.
//! GCS's bucket is the bucket-equivalent; conditional writes are native
//! (`x-goog-if-generation-match`) with no builder flag. The one twist vs.
//! S3/Azure: GCS keys conditional updates on the object *generation*, not
//! the HTTP ETag, so this provider carries the generation in
//! [`ObjectMeta::etag`] (an opaque version token) and returns it through
//! `UpdateVersion::version`.

use std::sync::Arc;

use object_store::gcp::GoogleCloudStorage;

/// GCS-backed `StorageProvider`. Cheap to clone; the inner
/// `GoogleCloudStorage` shares its HTTP client across clones.
#[derive(Debug)]
pub struct GcsStorageProvider {
    bucket: String,
    prefix: String,
    store: Arc<GoogleCloudStorage>,
}
