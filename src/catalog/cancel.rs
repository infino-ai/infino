// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`QueryCancel`]: the handle a caller keeps to stop a running
//! [`Connection::query_sql_cancellable`](crate::Connection::query_sql_cancellable).

use tokio_util::sync::CancellationToken;

use crate::InfinoError;

/// Stops a running SQL query from another thread.
///
/// Pass it to [`Connection::query_sql_cancellable`](crate::Connection::query_sql_cancellable)
/// and call [`cancel`](Self::cancel) from anywhere: the call returns
/// [`InfinoError::Cancelled`] soon after, and the work it drops gives back what
/// it reserved (that method lists what finishes regardless). Clones share one
/// state, so cancelling any clone cancels them all. Cancelling is permanent: a
/// handle that has fired cancels every query it is passed to.
///
/// [`InfinoError::Cancelled`]: crate::InfinoError::Cancelled
#[derive(Debug, Clone, Default)]
pub struct QueryCancel {
    token: CancellationToken,
}

impl QueryCancel {
    /// A handle that has not fired.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask every query holding this handle to stop. Idempotent.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether [`cancel`](Self::cancel) has been called on this handle or a clone.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// `Cancelled` once the handle has fired.
    pub(crate) fn check(&self) -> Result<(), InfinoError> {
        if self.is_cancelled() {
            return Err(cancelled());
        }
        Ok(())
    }

    /// Run `work` unless the handle fires first, in which case `work` is
    /// dropped and the answer is `Cancelled`. The handle is polled first, so
    /// once it has fired no result is taken, even one about to land.
    pub(crate) async fn unless_cancelled<T>(
        &self,
        work: impl Future<Output = Result<T, InfinoError>>,
    ) -> Result<T, InfinoError> {
        tokio::select! {
            biased;
            () = self.token.cancelled() => Err(cancelled()),
            result = work => result,
        }
    }
}

/// The error a cancelled query returns.
fn cancelled() -> InfinoError {
    InfinoError::Cancelled("the query was cancelled".to_string())
}
