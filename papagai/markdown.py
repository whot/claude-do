#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Markdown file parsing utilities."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

# Regex pattern for frontmatter key: value pairs
# Matches: key_name: value
KEY_VALUE_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+):\s*(.*)$")


@dataclass
class Markdown:
    """
    Markdown file with parsed frontmatter.

    Attributes:
        frontmatter: Dictionary of frontmatter key-value pairs
    """

    frontmatter: dict[str, str] = field(default_factory=dict)
    text: str = ""

    @classmethod
    def from_string(cls, content: str) -> Self:
        """
        Parse a markdown string and extract frontmatter.

        Frontmatter is delimited by --- at the start and end, and contains
        key: value pairs that may span multiple lines.

        Args:
            content: Markdown content as a string

        Returns:
            Markdown instance with parsed frontmatter
        """
        lines = content.split("\n")

        # Check if content starts with ---
        if not lines or lines[0].strip() != "---":
            return cls(frontmatter={}, text=content)

        # Find the closing ---
        current_key = None
        current_value = []

        frontmatter = {}
        text = content  # Default to full content if no closing --- found

        for idx, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                # End of frontmatter
                if current_key:
                    frontmatter[current_key] = "\n".join(current_value).strip()
                text = "\n".join(lines[idx + 1 :])
                break

            # Check if this is a key: value line using regex
            match = KEY_VALUE_PATTERN.match(line)
            if match:
                # Save previous key-value if exists
                if current_key:
                    frontmatter[current_key] = "\n".join(current_value).strip()

                # Start new key-value
                current_key = match.group(1)
                current_value = [match.group(2)]
            elif current_key:
                # Continuation of multi-line value
                current_value.append(line)
        else:
            # No closing --- found, reset to empty frontmatter
            frontmatter = {}
            text = content

        return cls(frontmatter=frontmatter, text=text)

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        """
        Parse a markdown file and extract frontmatter.

        Frontmatter is delimited by --- at the start and end, and contains
        key: value pairs that may span multiple lines.

        Args:
            file_path: Path to the markdown file

        Returns:
            Markdown instance with parsed frontmatter

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
        """
        content = file_path.read_text()
        return cls.from_string(content)


@dataclass
class MarkdownInstructions(Markdown):
    """
    Markdown file with parsed instructions frontmatter.

    Inherits from Markdown and adds convenience fields for description and tools.

    Attributes:
        description: Description from frontmatter, or empty string if not found
        tools: List of tool strings parsed from frontmatter, or empty list if not found
    """

    description: str = ""
    tools: list[str] = field(default_factory=list)

    @classmethod
    def from_string(cls, content: str) -> Self:
        """
        Parse a markdown string and extract instructions frontmatter.

        Parses the description and tools from the frontmatter.
        Tools are comma-separated, but commas inside parentheses/braces are preserved.

        Args:
            content: Markdown content as a string

        Returns:
            MarkdownInstructions instance with parsed frontmatter and fields
        """
        # First parse the base frontmatter
        base = Markdown.from_string(content)

        # Extract description
        description = base.frontmatter.get("description", "")

        # Extract and parse tools
        tools_str = base.frontmatter.get("tools", "")
        tools = cls._parse_tools(tools_str)

        return cls(
            frontmatter=base.frontmatter,
            text=base.text,
            description=description,
            tools=tools,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        """
        Parse a markdown file and extract instructions frontmatter.

        Parses the description and tools from the frontmatter.
        Tools are comma-separated, but commas inside parentheses/braces are preserved.

        Args:
            file_path: Path to the markdown file

        Returns:
            MarkdownInstructions instance with parsed frontmatter and fields

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
        """
        # First parse the base frontmatter
        base = Markdown.from_file(file_path)

        # Extract description
        description = base.frontmatter.get("description", "")

        # Extract and parse tools
        tools_str = base.frontmatter.get("tools", "")
        tools = cls._parse_tools(tools_str)

        return cls(
            frontmatter=base.frontmatter,
            text=base.text,
            description=description,
            tools=tools,
        )

    def combine(self, other: Self) -> Self:
        """
        Combine this MarkdownInstructions with another.

        Creates a new MarkdownInstructions with:
        - Text from both objects concatenated (self.text + other.text)
        - Tools from both objects combined (deduplicated)
        - Description from only the first object (self)
        - Frontmatter merged from both objects

        Args:
            other: Another MarkdownInstructions to combine with

        Returns:
            New MarkdownInstructions with combined content
        """
        # Combine text from both objects
        combined_text = self.text + "\n" + other.text

        # Combine tools, preserving order and removing duplicates
        seen = set()
        combined_tools = []
        for tool in self.tools + other.tools:
            if tool not in seen:
                seen.add(tool)
                combined_tools.append(tool)

        # Merge frontmatter (self takes precedence for duplicate keys)
        combined_frontmatter = {**other.frontmatter, **self.frontmatter}

        # Description only from first object
        description = self.description

        return type(self)(
            frontmatter=combined_frontmatter,
            text=combined_text,
            description=description,
            tools=combined_tools,
        )

    @staticmethod
    def _parse_tools(tools_str: str) -> list[str]:
        """
        Parse comma-separated tools, respecting nested parentheses and braces.

        Args:
            tools_str: Comma-separated tool specifications

        Returns:
            List of tool strings, or empty list if input is empty
        """
        if not tools_str:
            return []

        # Split by comma, but only if not inside parentheses or braces
        tools = []
        current_tool = []
        depth = 0  # Track nesting depth of () and {}

        for char in tools_str:
            if char in "({":
                depth += 1
                current_tool.append(char)
            elif char in ")}":
                depth -= 1
                current_tool.append(char)
            elif char == "," and depth == 0:
                # Comma at top level - end of current tool
                tool = "".join(current_tool).strip()
                if tool:
                    tools.append(tool)
                current_tool = []
            else:
                current_tool.append(char)

        # Add the last tool
        tool = "".join(current_tool).strip()
        if tool:
            tools.append(tool)

        return tools
