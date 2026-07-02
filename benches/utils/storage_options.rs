// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Object-store credentials gathered from the environment for real-cloud
//! bench runs. Infino's providers no longer read the environment; the
//! bench harness collects credentials here and passes them as config.

use std::collections::HashMap;

/// Collect options from `env_to_key` pairs that are set, mapping each
/// environment variable to its `object_store` config key.
fn storage_options_from_env(env_to_key: &[(&str, &str)]) -> HashMap<String, String> {
    env_to_key
        .iter()
        .filter_map(|(env, key)| std::env::var(env).ok().map(|v| (key.to_string(), v)))
        .collect()
}

/// Standard S3 credential options from the AWS environment.
/// `AWS_DEFAULT_REGION` is listed before `AWS_REGION` so the latter wins
/// when both are set (matching AWS precedence; equal keys, last wins).
pub fn s3_storage_options_from_env() -> HashMap<String, String> {
    storage_options_from_env(&[
        ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
        ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
        ("AWS_SESSION_TOKEN", "aws_session_token"),
        ("AWS_DEFAULT_REGION", "aws_region"),
        ("AWS_REGION", "aws_region"),
        ("AWS_ENDPOINT", "aws_endpoint"),
    ])
}

/// Standard Azure credential options from the environment.
pub fn azure_storage_options_from_env() -> HashMap<String, String> {
    storage_options_from_env(&[
        ("AZURE_STORAGE_ACCOUNT_NAME", "azure_storage_account_name"),
        ("AZURE_STORAGE_ACCOUNT_KEY", "azure_storage_account_key"),
    ])
}

/// Standard GCS credential options from the environment.
/// `GOOGLE_APPLICATION_CREDENTIALS` (a service-account key path) maps to
/// object_store's `google_service_account`.
pub fn gcs_storage_options_from_env() -> HashMap<String, String> {
    storage_options_from_env(&[
        ("GOOGLE_APPLICATION_CREDENTIALS", "google_service_account"),
        ("GOOGLE_SERVICE_ACCOUNT_KEY", "google_service_account_key"),
    ])
}
