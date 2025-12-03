"""Structured data extraction without agent overhead"""

from typing import Type, Union
from pydantic import BaseModel

from jetflow.clients.base import BaseClient, AsyncBaseClient


class Extract:
    """Extract structured data from text using a schema.

    Uses provider-native structured outputs for guaranteed schema compliance.

    Example:
        from pydantic import BaseModel
        from jetflow import Extract
        from jetflow.clients.openai import OpenAIClient

        class Person(BaseModel):
            name: str
            age: int

        client = OpenAIClient()
        extractor = Extract(client, Person)
        person = extractor.run("John is 25 years old")
        print(person.name, person.age)  # John 25
    """

    def __init__(
        self,
        client: BaseClient,
        schema: Type[BaseModel],
        system_prompt: str = "Extract the requested information.",
    ):
        """Initialize extractor.

        Args:
            client: Sync client instance (OpenAIClient, AnthropicClient, etc.)
            schema: Pydantic model defining the output structure
            system_prompt: Instructions for extraction
        """
        if isinstance(client, AsyncBaseClient):
            raise TypeError(
                "Extract requires a sync client. Use AsyncExtract for async clients."
            )
        self.client = client
        self.schema = schema
        self.system_prompt = system_prompt

    def run(self, query: str) -> BaseModel:
        """Extract structured data from the query.

        Args:
            query: Input text to extract from

        Returns:
            Parsed Pydantic model instance
        """
        return self.client.extract(self.schema, query, self.system_prompt)


class AsyncExtract:
    """Async version of Extract for use with async clients.

    Example:
        from pydantic import BaseModel
        from jetflow import AsyncExtract
        from jetflow.clients.openai import AsyncOpenAIClient

        class Person(BaseModel):
            name: str
            age: int

        client = AsyncOpenAIClient()
        extractor = AsyncExtract(client, Person)
        person = await extractor.run("John is 25 years old")
        print(person.name, person.age)  # John 25
    """

    def __init__(
        self,
        client: AsyncBaseClient,
        schema: Type[BaseModel],
        system_prompt: str = "Extract the requested information.",
    ):
        """Initialize async extractor.

        Args:
            client: Async client instance (AsyncOpenAIClient, AsyncAnthropicClient, etc.)
            schema: Pydantic model defining the output structure
            system_prompt: Instructions for extraction
        """
        if isinstance(client, BaseClient) and not isinstance(client, AsyncBaseClient):
            raise TypeError(
                "AsyncExtract requires an async client. Use Extract for sync clients."
            )
        self.client = client
        self.schema = schema
        self.system_prompt = system_prompt

    async def run(self, query: str) -> BaseModel:
        """Extract structured data from the query.

        Args:
            query: Input text to extract from

        Returns:
            Parsed Pydantic model instance
        """
        return await self.client.extract(self.schema, query, self.system_prompt)
