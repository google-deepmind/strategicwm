# Copyright 2025 The strategicwm Authors.
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

"""Tests prompting utils: json parsing, pretty printing, numeric substrings."""

from absl.testing import absltest
from absl.testing import parameterized
import pydantic

from strategicwm._src.se import prompts


class TestModel(pydantic.BaseModel):
  name: str
  value: int


class ParseJsonTest(absltest.TestCase):

  def test_parse_json_success(self):
    response = '```json\n{"name": "test", "value": 123}\n```'
    result = prompts.parse_json(response, TestModel)
    self.assertEqual(result, {"name": "test", "value": 123})

  def test_parse_json_malformed_json(self):
    response = '```json\n{"name": "test", "value": 123,}\n```'
    with self.assertRaisesRegex(ValueError, "Failed to load LLM JSON block"):
      prompts.parse_json(response, TestModel)

  def test_parse_json_pydantic_validation_error(self):
    response = '```json\n{"name": "test", "value": "not-an-int"}\n```'
    with self.assertRaisesRegex(
        ValueError, "LLM response failed pydantic validation"
    ):
      prompts.parse_json(response, TestModel)


class KeyValueListToStrTest(absltest.TestCase):

  def test_empty_list(self):
    self.assertEqual(prompts.key_value_list_to_str([]), "")

  def test_short_values(self):
    data = [("key1", "value1"), ("key2", "value2")]
    expected = "* key1: value1\n* key2: value2"
    self.assertEqual(prompts.key_value_list_to_str(data), expected)

  def test_long_value(self):
    vals = ["a" * 20 for _ in range(5)]
    long_value = "\n".join(vals)
    data = [("key1", long_value)]
    expected = "* key1:\n--> " + "\n--> ".join(vals)
    self.assertEqual(prompts.key_value_list_to_str(data, width=80), expected)

  def test_newline_value(self):
    data = [("key1", "value1\nvalue2")]
    expected = "* key1:\n--> value1\n--> value2"
    self.assertEqual(prompts.key_value_list_to_str(data), expected)


class RetrieveNumericBlockTest(parameterized.TestCase):

  @parameterized.parameters(
      ("abc", ""),
      ("123", "123"),
      ("abc 123 def", "123"),
      ("abc -123 def", "-123"),
      ("123a456", "123"),
      ("-123a456", "-123"),
  )
  def test_get_numeric_block(self, text, expected):
    self.assertEqual(prompts.get_numeric_block(text, 0), expected)


if __name__ == "__main__":
  absltest.main()
