// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Runtime-swappable object-store credentials.
//!
//! [`RotatingCredentialProvider`] backs `object_store`'s `with_credentials`
//! with an `ArcSwap`, so a worker can rotate a static key without rebuilding
//! the store.

use std::{fmt::Debug, str::FromStr, sync::Arc};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use object_store::{
    CredentialProvider,
    aws::{AmazonS3ConfigKey, AwsCredential, AwsCredentialProvider},
    azure::{AzureAccessKey, AzureConfigKey, AzureCredential, AzureCredentialProvider},
};

use super::{StorageError, StorageOptions};

#[derive(Debug)]
pub struct RotatingCredentialProvider<T> {
    current: ArcSwap<T>,
}

impl<T: Debug + Send + Sync> RotatingCredentialProvider<T> {
    pub(crate) fn new(initial: T) -> Self {
        Self {
            current: ArcSwap::from_pointee(initial),
        }
    }

    /// Publish a new credential; the next `get_credential` returns it.
    pub(crate) fn store(&self, next: T) {
        self.current.store(Arc::new(next));
    }
}

#[async_trait]
impl<T: Debug + Send + Sync> CredentialProvider for RotatingCredentialProvider<T> {
    type Credential = T;

    async fn get_credential(&self) -> object_store::Result<Arc<T>> {
        Ok(self.current.load_full())
    }
}

/// Swappable credential shared (cloned `Arc`) into every provider a
/// connection builds. Holds only the rotatable static credential.
#[derive(Debug, Clone)]
pub enum BackendCredentials {
    S3(Arc<RotatingCredentialProvider<AwsCredential>>),
    Azure(Arc<RotatingCredentialProvider<AzureCredential>>),
}

impl BackendCredentials {
    /// `Some` when `opts` carries static S3 credentials, else `None`
    /// (ambient identity / role — object_store refreshes those itself).
    pub(crate) fn s3_from_options(opts: &StorageOptions) -> Result<Option<Self>, StorageError> {
        Ok(aws_credential(opts)?.map(|c| Self::S3(Arc::new(RotatingCredentialProvider::new(c)))))
    }

    /// `Some` when `opts` carries a static Azure account key, else `None`.
    pub(crate) fn azure_from_options(opts: &StorageOptions) -> Result<Option<Self>, StorageError> {
        Ok(azure_credential(opts)?
            .map(|c| Self::Azure(Arc::new(RotatingCredentialProvider::new(c)))))
    }

    /// Swap in the credential parsed from `opts`. Errors (leaving the old
    /// credential live) if `opts` carries none for this backend.
    pub(crate) fn rotate(&self, opts: &StorageOptions) -> Result<(), StorageError> {
        match self {
            Self::S3(provider) => provider.store(aws_credential(opts)?.ok_or_else(no_credential)?),
            Self::Azure(provider) => {
                provider.store(azure_credential(opts)?.ok_or_else(no_credential)?)
            }
        }
        Ok(())
    }

    pub(crate) fn as_aws(&self) -> Option<AwsCredentialProvider> {
        match self {
            Self::S3(provider) => Some(Arc::clone(provider) as AwsCredentialProvider),
            Self::Azure(_) => None,
        }
    }

    pub(crate) fn as_azure(&self) -> Option<AzureCredentialProvider> {
        match self {
            Self::Azure(provider) => Some(Arc::clone(provider) as AzureCredentialProvider),
            Self::S3(_) => None,
        }
    }
}

/// Whether `key` is a rotatable static credential — kept out of
/// `with_config` so it can't shadow the rotating provider. Matches by
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

fn no_credential() -> StorageError {
    invalid("no rotatable credential in storage options for this backend")
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

    #[tokio::test]
    async fn get_credential_reflects_latest_store() {
        let provider = RotatingCredentialProvider::new("first".to_string());
        assert_eq!(*provider.get_credential().await.expect("cred"), "first");

        provider.store("second".to_string());
        assert_eq!(*provider.get_credential().await.expect("cred"), "second");
    }

    fn opts(pairs: &[(&str, &str)]) -> StorageOptions {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn s3_from_options_builds_or_skips() {
        let creds = BackendCredentials::s3_from_options(&opts(&[
            ("aws_access_key_id", "ak"),
            ("aws_secret_access_key", "sk"),
        ]))
        .expect("parse");
        let creds = creds.expect("some");
        assert!(creds.as_aws().is_some() && creds.as_azure().is_none());

        // Non-credential options alone → nothing to rotate.
        assert!(
            BackendCredentials::s3_from_options(&opts(&[("aws_region", "us-east-1")]))
                .expect("parse")
                .is_none()
        );
    }

    #[test]
    fn s3_partial_credentials_error() {
        assert!(
            BackendCredentials::s3_from_options(&opts(&[("aws_access_key_id", "ak")])).is_err()
        );
    }

    #[test]
    fn azure_account_key_validated() {
        // "a2V5" is valid base64; a bare "key" is not.
        assert!(
            BackendCredentials::azure_from_options(&opts(&[("azure_storage_account_key", "a2V5")]))
                .expect("parse")
                .is_some()
        );
        assert!(
            BackendCredentials::azure_from_options(&opts(&[(
                "azure_storage_account_key",
                "not-base64!"
            )]))
            .is_err()
        );
    }

    #[test]
    fn rotate_swaps_or_rejects() {
        let creds = BackendCredentials::s3_from_options(&opts(&[
            ("aws_access_key_id", "ak"),
            ("aws_secret_access_key", "sk"),
        ]))
        .expect("parse")
        .expect("some");

        assert!(
            creds
                .rotate(&opts(&[
                    ("aws_access_key_id", "ak2"),
                    ("aws_secret_access_key", "sk2"),
                ]))
                .is_ok()
        );
        // No credential in the new options → rejected, old stays live.
        assert!(creds.rotate(&opts(&[("aws_region", "eu-west-1")])).is_err());
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
