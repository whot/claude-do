#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for markdown parsing utilities."""

import logging
import pytest
from pathlib import Path
from papagai.markdown import Markdown

logger = logging.getLogger("papagai.test")


@pytest.fixture
def tmp_md_file(tmp_path):
    """Create a temporary markdown file for testing."""

    def _create_file(content: str) -> Path:
        md_file = tmp_path / "test.md"
        md_file.write_text(content)
        return md_file

    return _create_file


def test_parse_frontmatter_simple(tmp_md_file):
    """Test parsing simple frontmatter with single-line values."""
    content = """---
description: A simple description
author: John Doe
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter == {
        "description": "A simple description",
        "author": "John Doe",
    }
    assert md.text == "\n# Content here\n"


def test_parse_frontmatter_multiline_value(tmp_md_file):
    """Test parsing frontmatter with multi-line values."""
    content = """---
description: This is a long description
  that spans multiple lines
  and should be preserved
title: Short Title
---

# Content
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert "description" in md.frontmatter
    assert "that spans multiple lines" in md.frontmatter["description"]
    assert md.frontmatter["title"] == "Short Title"
    assert md.text == "\n# Content\n"


def test_parse_frontmatter_no_frontmatter(tmp_md_file):
    """Test file without frontmatter returns empty dict."""
    content = """# Just a regular markdown file

No frontmatter here.
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter == {}
    assert md.text == content


def test_parse_frontmatter_empty_file(tmp_md_file):
    """Test empty file returns empty dict."""
    content = ""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter == {}
    assert md.text == ""


def test_parse_frontmatter_only_opening_delimiter(tmp_md_file):
    """Test file with only opening --- delimiter."""
    content = """---
description: test
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    # Should return empty dict as there's no closing delimiter
    assert md.frontmatter == {}
    # Text should contain full content since frontmatter is incomplete
    assert md.text == content


def test_parse_frontmatter_colon_in_value(tmp_md_file):
    """Test frontmatter value containing colons."""
    content = """---
description: This is a description: with colons: in it
url: https://example.com:8080/path
---
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter["description"] == "This is a description: with colons: in it"
    assert md.frontmatter["url"] == "https://example.com:8080/path"


def test_parse_frontmatter_empty_values(tmp_md_file):
    """Test frontmatter with empty values."""
    content = """---
description:
author:
title: Has Value
---
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter["description"] == ""
    assert md.frontmatter["author"] == ""
    assert md.frontmatter["title"] == "Has Value"


def test_parse_frontmatter_whitespace_handling(tmp_md_file):
    """Test frontmatter with various whitespace."""
    content = """---
description:   leading and trailing spaces
author: normal
---
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    # Values should be stripped
    assert md.frontmatter["description"] == "leading and trailing spaces"
    assert md.frontmatter["author"] == "normal"


def test_parse_frontmatter_content_after_frontmatter(tmp_md_file):
    """Test that content after frontmatter is ignored."""
    content = """---
description: Test description
---

# This is markdown content

key: value that should not be parsed
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert md.frontmatter == {"description": "Test description"}
    assert "key" not in md.frontmatter
    assert (
        md.text
        == "\n# This is markdown content\n\nkey: value that should not be parsed\n"
    )


def test_parse_frontmatter_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    non_existent = Path("/tmp/non_existent_file_12345.md")

    with pytest.raises(FileNotFoundError):
        Markdown.from_file(non_existent)


def test_parse_frontmatter_real_example(tmp_md_file):
    """Test with a real example from the codebase."""
    content = """---
description: replace all literal floats/doubles written as integers to floating point (e.g. 1.0f/1.0)
---

You are a Software Developer with many years of experience in writing C code.
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert "description" in md.frontmatter
    assert "1.0f/1.0" in md.frontmatter["description"]
    assert (
        md.text
        == "\nYou are a Software Developer with many years of experience in writing C code.\n"
    )


def test_parse_frontmatter_multiple_keys(tmp_md_file):
    """Test frontmatter with multiple keys."""
    content = """---
title: Test Title
description: Test Description
author: Test Author
version: 1.0
tags: python, testing
---
"""
    md_file = tmp_md_file(content)
    md = Markdown.from_file(md_file)

    assert len(md.frontmatter) == 5
    assert md.frontmatter["title"] == "Test Title"
    assert md.frontmatter["description"] == "Test Description"
    assert md.frontmatter["author"] == "Test Author"
    assert md.frontmatter["version"] == "1.0"
    assert md.frontmatter["tags"] == "python, testing"


# Tests for Markdown.from_string


def test_markdown_from_string_simple():
    """Test Markdown.from_string with simple frontmatter."""
    content = """---
description: A simple description
author: John Doe
---

# Content here
"""
    md = Markdown.from_string(content)

    assert md.frontmatter == {
        "description": "A simple description",
        "author": "John Doe",
    }
    assert md.text == "\n# Content here\n"


def test_markdown_from_string_no_frontmatter():
    """Test Markdown.from_string without frontmatter."""
    content = """# Just a regular markdown file

No frontmatter here.
"""
    md = Markdown.from_string(content)

    assert md.frontmatter == {}
    assert md.text == content


def test_markdown_from_string_multiline_value():
    """Test Markdown.from_string with multi-line values."""
    content = """---
description: This is a long description
  that spans multiple lines
  and should be preserved
title: Short Title
---

# Content
"""
    md = Markdown.from_string(content)

    assert "description" in md.frontmatter
    assert "that spans multiple lines" in md.frontmatter["description"]
    assert md.frontmatter["title"] == "Short Title"
    assert md.text == "\n# Content\n"


def test_markdown_from_string_empty():
    """Test Markdown.from_string with empty string."""
    content = ""
    md = Markdown.from_string(content)

    assert md.frontmatter == {}
    assert md.text == ""


def test_markdown_from_string_no_closing_delimiter():
    """Test Markdown.from_string with no closing delimiter."""
    content = """---
description: test
"""
    md = Markdown.from_string(content)

    # Should return empty dict as there's no closing delimiter
    assert md.frontmatter == {}
    # Text should contain full content since frontmatter is incomplete
    assert md.text == content


def test_markdown_from_string_colon_in_value():
    """Test Markdown.from_string with colons in values."""
    content = """---
description: This is a description: with colons: in it
url: https://example.com:8080/path
---
"""
    md = Markdown.from_string(content)

    assert md.frontmatter["description"] == "This is a description: with colons: in it"
    assert md.frontmatter["url"] == "https://example.com:8080/path"


# Tests for MarkdownInstructions


def test_markdown_instructions_no_frontmatter(tmp_md_file):
    """Test MarkdownInstructions with no frontmatter."""
    from papagai.markdown import MarkdownInstructions

    content = "# Just a regular markdown file\n\nNo frontmatter here."
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.description == ""
    assert md.tools == []
    assert md.text == content


def test_markdown_instructions_no_tools_key(tmp_md_file):
    """Test MarkdownInstructions with frontmatter but no tools key."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
author: Test Author
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.description == "Test description"
    assert md.tools == []


def test_markdown_instructions_empty_tools_value(tmp_md_file):
    """Test MarkdownInstructions with empty tools value."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools:
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.description == "Test description"
    assert md.tools == []


def test_markdown_instructions_single_tool(tmp_md_file):
    """Test MarkdownInstructions with single tool."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(npm:*)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.description == "Test description"
    assert md.tools == ["Bash(npm:*)"]
    assert md.text == "\n# Content here\n"


def test_markdown_instructions_multiple_tools(tmp_md_file):
    """Test MarkdownInstructions with multiple tools."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(npm:*), Read(*.js), Write(*.ts)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.description == "Test description"
    assert md.tools == ["Bash(npm:*)", "Read(*.js)", "Write(*.ts)"]


def test_markdown_instructions_tools_with_whitespace(tmp_md_file):
    """Test MarkdownInstructions tools parsing with various whitespace."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(npm:*)  ,  Read(*.js)  ,Write(*.ts)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.tools == ["Bash(npm:*)", "Read(*.js)", "Write(*.ts)"]


def test_markdown_instructions_tools_skips_empty_entries(tmp_md_file):
    """Test MarkdownInstructions parsing skips empty entries between commas."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(npm:*), , Read(*.js),  ,Write(*.ts)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.tools == ["Bash(npm:*)", "Read(*.js)", "Write(*.ts)"]


def test_markdown_instructions_tools_complex_patterns(tmp_md_file):
    """Test MarkdownInstructions parsing tools with complex patterns."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(git:*), Glob(**/*.py), Grep(*.{js,ts}), Edit(./**/*)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.tools == [
        "Bash(git:*)",
        "Glob(**/*.py)",
        "Grep(*.{js,ts})",
        "Edit(./**/*)",
    ]


def test_markdown_instructions_tools_with_colons(tmp_md_file):
    """Test MarkdownInstructions parsing tools with colons in the tool specification."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(git:*), WebFetch(https://example.com)
---

# Content here
"""
    md_file = tmp_md_file(content)
    md = MarkdownInstructions.from_file(md_file)

    assert md.tools == ["Bash(git:*)", "WebFetch(https://example.com)"]


# Tests for MarkdownInstructions.from_string


def test_markdown_instructions_from_string_simple():
    """Test MarkdownInstructions.from_string with simple frontmatter."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: A simple description
tools: Bash(npm:*)
---

# Content here
"""
    md = MarkdownInstructions.from_string(content)

    assert md.description == "A simple description"
    assert md.tools == ["Bash(npm:*)"]
    assert md.text == "\n# Content here\n"


def test_markdown_instructions_from_string_no_frontmatter():
    """Test MarkdownInstructions.from_string with no frontmatter."""
    from papagai.markdown import MarkdownInstructions

    content = "# Just a regular markdown file\n\nNo frontmatter here."
    md = MarkdownInstructions.from_string(content)

    assert md.description == ""
    assert md.tools == []
    assert md.text == content


def test_markdown_instructions_from_string_multiple_tools():
    """Test MarkdownInstructions.from_string with multiple tools."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(npm:*), Read(*.js), Write(*.ts)
---

# Content here
"""
    md = MarkdownInstructions.from_string(content)

    assert md.description == "Test description"
    assert md.tools == ["Bash(npm:*)", "Read(*.js)", "Write(*.ts)"]


def test_markdown_instructions_from_string_empty():
    """Test MarkdownInstructions.from_string with empty string."""
    from papagai.markdown import MarkdownInstructions

    content = ""
    md = MarkdownInstructions.from_string(content)

    assert md.description == ""
    assert md.tools == []
    assert md.text == ""


def test_markdown_instructions_from_string_multiline_description():
    """Test MarkdownInstructions.from_string with multiline description."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: This is a long description
  that spans multiple lines
  and should be preserved
tools: Bash(git:*)
---

# Content
"""
    md = MarkdownInstructions.from_string(content)

    assert "that spans multiple lines" in md.description
    assert md.tools == ["Bash(git:*)"]


def test_markdown_instructions_from_string_no_closing_delimiter():
    """Test MarkdownInstructions.from_string with no closing delimiter."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: test
tools: Bash(npm:*)
"""
    md = MarkdownInstructions.from_string(content)

    # Should return empty frontmatter as there's no closing delimiter
    assert md.description == ""
    assert md.tools == []
    assert md.text == content


def test_markdown_instructions_from_string_complex_tools():
    """Test MarkdownInstructions.from_string with complex tool patterns."""
    from papagai.markdown import MarkdownInstructions

    content = """---
description: Test description
tools: Bash(git:*), Glob(**/*.py), Grep(*.{js,ts}), Edit(./**/*)
---

# Content here
"""
    md = MarkdownInstructions.from_string(content)

    assert md.tools == [
        "Bash(git:*)",
        "Glob(**/*.py)",
        "Grep(*.{js,ts})",
        "Edit(./**/*)",
    ]


# Tests for MarkdownInstructions.combine


def test_markdown_instructions_combine_simple():
    """Test combining two MarkdownInstructions objects."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
description: First description
tools: Bash(npm:*)
---

First text.
""")

    second = MarkdownInstructions.from_string("""---
description: Second description
tools: Read(*.js)
---

Second text.
""")

    combined = first.combine(second)

    assert combined.description == "First description"
    assert combined.text == "\nFirst text.\n\n\nSecond text.\n"
    assert combined.tools == ["Bash(npm:*)", "Read(*.js)"]


def test_markdown_instructions_combine_duplicate_tools():
    """Test combining with duplicate tools deduplicates them."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
description: First description
tools: Bash(npm:*), Read(*.js)
---

First text.
""")

    second = MarkdownInstructions.from_string("""---
description: Second description
tools: Read(*.js), Write(*.ts)
---

Second text.
""")

    combined = first.combine(second)

    assert combined.description == "First description"
    assert combined.tools == ["Bash(npm:*)", "Read(*.js)", "Write(*.ts)"]
    assert combined.text == "\nFirst text.\n\n\nSecond text.\n"


def test_markdown_instructions_combine_empty_tools():
    """Test combining when one object has no tools."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
description: First description
tools: Bash(npm:*)
---

First text.
""")

    second = MarkdownInstructions.from_string("""---
description: Second description
---

Second text.
""")

    combined = first.combine(second)

    assert combined.description == "First description"
    assert combined.tools == ["Bash(npm:*)"]
    assert combined.text == "\nFirst text.\n\n\nSecond text.\n"


def test_markdown_instructions_combine_no_frontmatter():
    """Test combining when objects have no frontmatter."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("First text.")
    second = MarkdownInstructions.from_string("Second text.")

    combined = first.combine(second)

    assert combined.description == ""
    assert combined.tools == []
    assert combined.text == "First text.\nSecond text."


def test_markdown_instructions_combine_frontmatter_merge():
    """Test that frontmatter is merged correctly."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
description: First description
author: First Author
version: 1.0
---

First text.
""")

    second = MarkdownInstructions.from_string("""---
description: Second description
author: Second Author
title: Second Title
---

Second text.
""")

    combined = first.combine(second)

    # First object's frontmatter takes precedence
    assert combined.frontmatter["description"] == "First description"
    assert combined.frontmatter["author"] == "First Author"
    assert combined.frontmatter["version"] == "1.0"
    # Second object's unique keys are included
    assert combined.frontmatter["title"] == "Second Title"


def test_markdown_instructions_combine_preserves_tool_order():
    """Test that tool order is preserved from first then second."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
tools: Bash(git:*), Glob(**/*.py), Read(*.js)
---

First text.
""")

    second = MarkdownInstructions.from_string("""---
tools: Write(*.ts), Edit(*.md)
---

Second text.
""")

    combined = first.combine(second)

    assert combined.tools == [
        "Bash(git:*)",
        "Glob(**/*.py)",
        "Read(*.js)",
        "Write(*.ts)",
        "Edit(*.md)",
    ]


def test_markdown_instructions_combine_empty_text():
    """Test combining when one object has empty text."""
    from papagai.markdown import MarkdownInstructions

    first = MarkdownInstructions.from_string("""---
description: First description
tools: Bash(npm:*)
---
""")

    second = MarkdownInstructions.from_string("""---
description: Second description
tools: Read(*.js)
---

Second text.
""")

    combined = first.combine(second)

    assert combined.description == "First description"
    assert combined.text == "\n\nSecond text.\n"
    assert combined.tools == ["Bash(npm:*)", "Read(*.js)"]
