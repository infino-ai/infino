// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident fine-centroid blob for hidden SPFresh routing.

use std::sync::Arc;

use bytes::Bytes;

use crate::{
    storage::StorageProvider,
    supertable::manifest::{Manifest, part::ContentHash},
};

/// Magic prefix on a hidden SPFresh fine-centroid blob.
const CENTROIDS_MAGIC: &[u8; 4] = b"HCEN";
/// Wire-format version for hidden SPFresh fine-centroid blobs.
const CENTROIDS_VERSION: u8 = 1;
/// Bytes in a little-endian `u32` header word.
const U32_BYTES: usize = 4;
/// Bytes in a little-endian `f32` centroid component.
const F32_BYTES: usize = 4;
/// Header: magic (4) + version (1) + dim (4) + n_centroids (4).
const CENTROIDS_HEADER_LEN: usize = CENTROIDS_MAGIC.len() + 1 + U32_BYTES + U32_BYTES;

#[derive(Debug, Clone, Default)]
pub(crate) struct ResidentCentroids {
    pub(crate) dim: usize,
    pub(crate) centroids: Arc<[f32]>,
}

impl ResidentCentroids {
    pub(crate) fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    pub(crate) fn centroid(&self, cluster_id: u32) -> Option<&[f32]> {
        let start = cluster_id as usize * self.dim;
        let end = start.checked_add(self.dim)?;
        self.centroids.get(start..end)
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HiddenCentroidsError {
    #[error("centroid blob truncated")]
    Truncated,
    #[error("centroid blob bad magic")]
    BadMagic,
    #[error("centroid blob unsupported version {0}")]
    UnsupportedVersion(u8),
    #[error("centroid blob dimension mismatch")]
    DimMismatch,
    #[error("storage: {0}")]
    Storage(String),
    #[error("content hash mismatch")]
    HashMismatch,
}

pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("spfresh-centroids/centroids-{}.bin", hash.to_hex())
}

pub(crate) fn encode_centroids(
    dim: usize,
    centroids: &[f32],
) -> Result<Vec<u8>, HiddenCentroidsError> {
    if dim == 0 || centroids.len() % dim != 0 {
        return Err(HiddenCentroidsError::DimMismatch);
    }
    let n_cent = centroids.len() / dim;
    let mut out = Vec::with_capacity(CENTROIDS_HEADER_LEN + centroids.len() * F32_BYTES);
    out.extend_from_slice(CENTROIDS_MAGIC);
    out.push(CENTROIDS_VERSION);
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(n_cent as u32).to_le_bytes());
    for value in centroids {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

pub(crate) fn decode_centroids(bytes: &[u8]) -> Result<ResidentCentroids, HiddenCentroidsError> {
    if bytes.len() < CENTROIDS_HEADER_LEN {
        return Err(HiddenCentroidsError::Truncated);
    }
    if &bytes[0..CENTROIDS_MAGIC.len()] != CENTROIDS_MAGIC {
        return Err(HiddenCentroidsError::BadMagic);
    }
    let version = bytes[CENTROIDS_MAGIC.len()];
    if version != CENTROIDS_VERSION {
        return Err(HiddenCentroidsError::UnsupportedVersion(version));
    }
    let dim_offset = CENTROIDS_MAGIC.len() + 1;
    let n_cent_offset = dim_offset + U32_BYTES;
    let dim = u32::from_le_bytes(
        bytes[dim_offset..dim_offset + U32_BYTES]
            .try_into()
            .map_err(|_| HiddenCentroidsError::Truncated)?,
    ) as usize;
    let n_cent = u32::from_le_bytes(
        bytes[n_cent_offset..n_cent_offset + U32_BYTES]
            .try_into()
            .map_err(|_| HiddenCentroidsError::Truncated)?,
    ) as usize;
    let body = &bytes[CENTROIDS_HEADER_LEN..];
    if dim == 0 || body.len() != n_cent * dim * F32_BYTES {
        return Err(HiddenCentroidsError::DimMismatch);
    }
    let mut centroids = Vec::with_capacity(n_cent * dim);
    for chunk in body.chunks_exact(F32_BYTES) {
        centroids.push(f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| HiddenCentroidsError::Truncated)?,
        ));
    }
    Ok(ResidentCentroids {
        dim,
        centroids: Arc::from(centroids),
    })
}

pub(crate) async fn load_resident_centroids(
    manifest: &Manifest,
    storage: &dyn StorageProvider,
) -> Result<ResidentCentroids, HiddenCentroidsError> {
    let Some((path, expected)) = manifest.spfresh_centroid_blob() else {
        return Ok(ResidentCentroids::default());
    };
    let bytes = fetch_and_verify(storage, &path, &expected).await?;
    decode_centroids(&bytes)
}

async fn fetch_and_verify(
    storage: &dyn StorageProvider,
    path: &str,
    expected: &ContentHash,
) -> Result<Bytes, HiddenCentroidsError> {
    let (bytes, _) = storage
        .get(path)
        .await
        .map_err(|e| HiddenCentroidsError::Storage(e.to_string()))?;
    if ContentHash::of(bytes.as_ref()) != *expected {
        return Err(HiddenCentroidsError::HashMismatch);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_blob_roundtrips() {
        let centroids = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes = encode_centroids(3, &centroids).expect("encode");
        let decoded = decode_centroids(&bytes).expect("decode");
        assert_eq!(decoded.dim, 3);
        assert_eq!(decoded.centroid(1), Some(&[3.0, 4.0, 5.0][..]));
    }
}
