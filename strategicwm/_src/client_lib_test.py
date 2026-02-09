# Copyright 2026 The strategicwm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests retriable error handling for client_lib."""

from absl.testing import absltest
from googleapiclient import errors

from strategicwm._src import client_lib


class MockResponse:

  def __init__(self, status=None):
    if status:
      self.status = status
    self.reason = "Mock Reason"


class HttpErrorRetriableTest(absltest.TestCase):

  def test_retriable_error_creation_success(self):
    for status in client_lib.HttpErrorRetriable.retriable_codes:
      mock_resp = MockResponse(status=status)
      http_error = errors.HttpError(mock_resp, b"error content")
      retriable_error = client_lib.HttpErrorRetriable(http_error)
      self.assertEqual(retriable_error.status, status)
      self.assertIn(
          f"HttpErrorRetriable (Status {status})", str(retriable_error)
      )

  def test_retriable_error_creation_fail_non_retriable(self):
    retriable_codes = client_lib.HttpErrorRetriable.retriable_codes
    status = retriable_codes[0]
    for _ in range(len(retriable_codes)):
      if status in retriable_codes:
        status += 1
      else:
        break
    mock_resp = MockResponse(status=status)
    http_error = errors.HttpError(mock_resp, b"error content")
    with self.assertRaisesRegex(
        ValueError,
        "HttpErrorRetriable cannot be created for status"
        f" {status}",
    ):
      client_lib.HttpErrorRetriable(http_error)

  def test_retriable_error_creation_fail_no_status(self):
    mock_resp = MockResponse()
    http_error = errors.HttpError(mock_resp, b"error content")
    with self.assertRaisesRegex(
        ValueError, "Original error object must have 'resp.status' attribute."
    ):
      client_lib.HttpErrorRetriable(http_error)


if __name__ == "__main__":
  absltest.main()
