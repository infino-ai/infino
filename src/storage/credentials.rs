// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Runtime-swappable storage options.
//!
//! [`SharedStorageOptions`] is a connection-scoped `ArcSwap<StorageOptions>`
//! shared (cloned `Arc`) into every provider a connection builds. The
//! credential providers it hands to `object_store`'s `with_credentials` read
//! the latest map on each request, so a worker can update a static key without
//! rebuilding any store.

use std::{fmt::Debug, str::FromStr, sync::Arc};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use object_store::{
    CredentialProvider, Error as ObjError,
    aws::{AmazonS3ConfigKey, AwsCredential, AwsCredentialProvider},
    azure::{AzureAccessKey, AzureConfigKey, AzureCredential, AzureCredentialProvider},
};

use super::{StorageError, StorageOptions};

/// Re-extracts its credential from the shared cell on every request, so a
/// swap of the cell takes effect without rebuilding the store.
#[derive(Debug)]
struct OptionsCredentialProvider<T> {
    options: Arc<ArcSwap<StorageOptions>>,
    /// Pulls the backend's credential out of the current map (`None` if absent).
    extract: fn(&StorageOptions) -> Result<Option<T>, StorageError>,
}

#[async_trait]
impl<T: Debug + Send + Sync> CredentialProvider for OptionsCredentialProvider<T> {
    type Credential = T;

    async fn get_credential(&self) -> object_store::Result<Arc<T>> {
        let options = self.options.load();
        match (self.extract)(&options) {
            Ok(Some(cred)) => Ok(Arc::new(cred)),
            Ok(None) => Err(cred_error("storage options no longer carry a credential")),
            Err(e) => Err(cred_error(&e.to_string())),
        }
    }
}

/// Connection-scoped storage options that can be updated at runtime.
///
/// Cloning shares the same cell (one `Arc`), so an [`Self::update`] on any
/// clone is seen by every provider built from it. Only credential keys take
/// effect on already-built stores (via the credential providers below);
/// non-credential keys apply to stores built after the update.
#[derive(Debug, Clone)]
pub(crate) struct SharedStorageOptions {
    options: Arc<ArcSwap<StorageOptions>>,
}

impl SharedStorageOptions {
    pub(crate) fn new(options: StorageOptions) -> Self {
        Self {
            options: Arc::new(ArcSwap::from_pointee(options)),
        }
    }

    /// Current options, for building a store's `with_config`.
    pub(crate) fn snapshot(&self) -> Arc<StorageOptions> {
        self.options.load_full()
    }

    /// Merge `patch` over the current options and publish the result. Rejects
    /// (leaving the old options live) if the merged view carries a malformed
    /// or incomplete credential.
    pub(crate) fn update(&self, patch: &StorageOptions) -> Result<(), StorageError> {
        let mut merged = self.options.load().as_ref().clone();
        merged.extend(patch.iter().map(|(k, v)| (k.clone(), v.clone())));
        // Validate both backends' credentials parse before publishing.
        aws_credential(&merged)?;
        azure_credential(&merged)?;
        self.options.store(Arc::new(merged));
        Ok(())
    }

    /// An S3 credential provider reading this cell, or `None` when the options
    /// carry no static S3 credential (ambient identity — object_store refreshes
    /// that itself). `Err` if the credential is present but incomplete.
    pub(crate) fn aws_provider(&self) -> Result<Option<AwsCredentialProvider>, StorageError> {
        Ok(aws_credential(&self.options.load())?.map(|_| {
            Arc::new(OptionsCredentialProvider {
                options: Arc::clone(&self.options),
                extract: aws_credential,
            }) as AwsCredentialProvider
        }))
    }

    /// An Azure credential provider reading this cell, or `None` when the
    /// options carry no static account key.
    pub(crate) fn azure_provider(&self) -> Result<Option<AzureCredentialProvider>, StorageError> {
        Ok(azure_credential(&self.options.load())?.map(|_| {
            Arc::new(OptionsCredentialProvider {
                options: Arc::clone(&self.options),
                extract: azure_credential,
            }) as AzureCredentialProvider
        }))
    }
}

/// Whether `key` is a rotatable static credential — kept out of
/// `with_config` so it can't shadow the credential provider. Matches by
/// config-key variant, so aliases are covered too.
pub(crate) fn is_s3_credential_key(key: &str) -> bool {
    matches!(
        AmazonS3ConfigKey::from_str(key),
        Ok(AmazonS3ConfigKey::AccessKeyId
            | AmazonS3ConfigKey::SecretAccessKey
            | AmazonS3ConfigKey::Token)
    )
}

pub(crate) fn is_azure_credential_key(key: &str) -> bool {
    matches!(AzureConfigKey::from_str(key), Ok(AzureConfigKey::AccessKey))
}

fn aws_credential(opts: &StorageOptions) -> Result<Option<AwsCredential>, StorageError> {
    let (mut key_id, mut secret_key, mut token) = (None, None, None);
    for (key, value) in opts {
        match AmazonS3ConfigKey::from_str(key) {
            Ok(AmazonS3ConfigKey::AccessKeyId) => key_id = Some(value.clone()),
            Ok(AmazonS3ConfigKey::SecretAccessKey) => secret_key = Some(value.clone()),
            Ok(AmazonS3ConfigKey::Token) => token = Some(value.clone()),
            _ => {}
        }
    }
    match (key_id, secret_key) {
        (Some(key_id), Some(secret_key)) => Ok(Some(AwsCredential {
            key_id,
            secret_key,
            token,
        })),
        (None, None) => Ok(None),
        _ => Err(invalid(
            "s3 needs both aws_access_key_id and aws_secret_access_key",
        )),
    }
}

fn azure_credential(opts: &StorageOptions) -> Result<Option<AzureCredential>, StorageError> {
    for (key, value) in opts {
        if matches!(AzureConfigKey::from_str(key), Ok(AzureConfigKey::AccessKey)) {
            let access_key = AzureAccessKey::try_new(value)
                .map_err(|e| invalid(&format!("invalid azure_storage_account_key: {e}")))?;
            return Ok(Some(AzureCredential::AccessKey(access_key)));
        }
    }
    Ok(None)
}

fn cred_error(msg: &str) -> ObjError {
    ObjError::Generic {
        store: "credentials",
        source: msg.to_string().into(),
    }
}

fn invalid(msg: &str) -> StorageError {
    StorageError::Permanent {
        uri: "credentials".to_string(),
        source: msg.to_string().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> StorageOptions {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn aws_provider_present_only_with_static_credentials() {
        let cell = SharedStorageOptions::new(opts(&[
            ("aws_access_key_id", "ak"),
            ("aws_secret_access_key", "sk"),
        ]));
        assert!(cell.aws_provider().expect("parse").is_some());
        assert!(cell.azure_provider().expect("parse").is_none());

        // Non-credential options alone → ambient identity, no provider.
        let ambient = SharedStorageOptions::new(opts(&[("aws_region", "us-east-1")]));
        assert!(ambient.aws_provider().expect("parse").is_none());
    }

    #[test]
    fn partial_s3_credentials_error() {
        let cell = SharedStorageOptions::new(opts(&[("aws_access_key_id", "ak")]));
        assert!(cell.aws_provider().is_err());
    }

    #[test]
    fn azure_account_key_validated() {
        // "a2V5" is valid base64; a bare "key" is not.
        let ok = SharedStorageOptions::new(opts(&[("azure_storage_account_key", "a2V5")]));
        assert!(ok.azure_provider().expect("parse").is_some());

        let bad = SharedStorageOptions::new(opts(&[("azure_storage_account_key", "not-base64!")]));
        assert!(bad.azure_provider().is_err());
    }

    #[tokio::test]
    async fn get_credential_reflects_latest_update() {
        let cell = SharedStorageOptions::new(opts(&[
            ("aws_access_key_id", "ak"),
            ("aws_secret_access_key", "sk"),
        ]));
        let provider = cell.aws_provider().expect("parse").expect("some");
        assert_eq!(provider.get_credential().await.expect("cred").key_id, "ak");

        cell.update(&opts(&[("aws_access_key_id", "ak2")]))
            .expect("update");
        // Merge keeps the old secret; the swapped access key is now served.
        assert_eq!(provider.get_credential().await.expect("cred").key_id, "ak2");
    }

    #[test]
    fn update_merges_and_rejects_malformed() {
        let cell = SharedStorageOptions::new(opts(&[
            ("aws_access_key_id", "ak"),
            ("aws_secret_access_key", "sk"),
        ]));
        // A non-credential key merges cleanly (takes effect on future stores).
        assert!(cell.update(&opts(&[("aws_region", "eu-west-1")])).is_ok());
        // A malformed azure key is rejected; the old options stay live.
        assert!(
            cell.update(&opts(&[("azure_storage_account_key", "not-base64!")]))
                .is_err()
        );
        assert_eq!(
            cell.snapshot().get("aws_access_key_id").map(String::as_str),
            Some("ak")
        );
    }

    #[test]
    fn credential_key_classification() {
        assert!(is_s3_credential_key("aws_secret_access_key"));
        assert!(is_s3_credential_key("session_token")); // alias
        assert!(!is_s3_credential_key("aws_region"));
        assert!(is_azure_credential_key("account_key")); // alias
        assert!(!is_azure_credential_key("azure_storage_account_name"));
    }
}
