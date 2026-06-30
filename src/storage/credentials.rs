// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Runtime-swappable object-store credentials.
//!
//! [`RotatingCredentialProvider`] backs `object_store`'s `with_credentials`
//! with an `ArcSwap`, so a worker can rotate a static key without rebuilding
//! the store.

use std::{fmt::Debug, sync::Arc};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use object_store::CredentialProvider;

#[derive(Debug)]
pub(crate) struct RotatingCredentialProvider<T> {
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
}
