"""Tests for Message serialization to/from database"""

import pytest
from jetflow.models.message import Message, TextBlock, ActionBlock, ThoughtBlock


class TestMessageBlocks:
    """Test blocks-based message handling"""

    def test_simple_message_no_interleaving(self):
        """Simple text + action message should not have interleaving"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Hello'))
        msg.blocks.append(ActionBlock(id='a1', name='test', status='completed', body={}))

        assert msg.has_interleaving is False
        assert msg.content == 'Hello'
        assert len(msg.actions) == 1

    def test_web_search_has_interleaving(self):
        """Message with web search should have interleaving"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Searching...'))
        msg.blocks.append(ActionBlock(id='ws1', name='web_search', status='completed', body={}, server_executed=True))
        msg.blocks.append(TextBlock(text='Found results'))

        assert msg.has_interleaving is True

    def test_server_executed_action_has_interleaving(self):
        """Message with server-executed action should have interleaving"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(ActionBlock(id='ws1', name='web_search', status='completed', body={}, server_executed=True))

        assert msg.has_interleaving is True

    def test_text_action_text_has_interleaving(self):
        """Text-Action-Text pattern should have interleaving"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Before'))
        msg.blocks.append(ActionBlock(id='a1', name='test', status='completed', body={}))
        msg.blocks.append(TextBlock(text='After'))

        assert msg.has_interleaving is True

    def test_action_mutation_persists(self):
        """Mutating action via .actions property should persist to blocks"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(ActionBlock(id='a1', name='test', status='completed', body={}))

        msg.actions[0].result = {'foo': 'bar'}
        assert msg.blocks[0].result == {'foo': 'bar'}


class TestMessageDbSerialization:
    """Test database serialization"""

    def test_to_db_row_simple(self):
        """Simple message serializes without blocks column"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Hello'))

        row = msg.to_db_row(session_id='sess1', user_id='user1')

        assert row['content'] == 'Hello'
        assert row['session_id'] == 'sess1'
        assert row['user_id'] == 'user1'
        assert row['role'] == 'Assistant'
        assert row['status'] == 'Completed'
        assert 'blocks' not in row

    def test_to_db_row_with_actions(self):
        """Message with actions serializes actions column"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Hello'))
        msg.blocks.append(ActionBlock(id='a1', name='get_data', status='completed', body={'q': 'test'}))

        row = msg.to_db_row()

        assert row['content'] == 'Hello'
        assert len(row['actions']) == 1
        assert row['actions'][0]['name'] == 'get_data'
        assert 'blocks' not in row

    def test_to_db_row_with_thought(self):
        """Message with thought serializes thought column"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(ThoughtBlock(id='t1', summaries=['Thinking...']))
        msg.blocks.append(TextBlock(text='Result'))

        row = msg.to_db_row()

        assert row['thought'] is not None
        assert row['thought']['summaries'] == ['Thinking...']

    def test_to_db_row_interleaved_includes_blocks(self):
        """Interleaved message includes blocks column"""
        msg = Message(role='assistant', status='completed')
        msg.blocks.append(TextBlock(text='Before'))
        msg.blocks.append(ActionBlock(id='ws1', name='web_search', status='completed', body={}, result={'results': [{'title': 'Result'}]}, server_executed=True))
        msg.blocks.append(TextBlock(text='After'))

        row = msg.to_db_row()

        assert 'blocks' in row
        assert len(row['blocks']) == 3
        assert row['blocks'][0]['type'] == 'text'
        assert row['blocks'][1]['type'] == 'action'
        assert row['blocks'][1]['server_executed'] is True
        assert row['blocks'][2]['type'] == 'text'

    def test_from_db_row_legacy(self):
        """Deserialize legacy row without blocks"""
        row = {
            'id': 'msg1',
            'role': 'Assistant',
            'status': 'Completed',
            'content': 'Hello',
            'actions': [{'id': 'a1', 'name': 'test', 'status': 'completed', 'body': {}}],
            'thought': None,
            'citations': None,
            'sources': None,
        }

        msg = Message.from_db_row(row)

        assert msg.id == 'msg1'
        assert msg.role == 'assistant'
        assert msg.content == 'Hello'
        assert len(msg.actions) == 1
        assert len(msg.blocks) == 2  # TextBlock + ActionBlock

    def test_from_db_row_with_blocks(self):
        """Deserialize row with blocks column"""
        row = {
            'id': 'msg2',
            'role': 'Assistant',
            'status': 'Completed',
            'blocks': [
                {'type': 'text', 'text': 'Before'},
                {'type': 'action', 'id': 'ws1', 'name': 'web_search', 'status': 'completed', 'body': {}, 'result': {'results': []}, 'server_executed': True},
                {'type': 'text', 'text': 'After'},
            ],
        }

        msg = Message.from_db_row(row)

        assert len(msg.blocks) == 3
        assert isinstance(msg.blocks[0], TextBlock)
        assert isinstance(msg.blocks[1], ActionBlock)
        assert msg.blocks[1].server_executed is True
        assert isinstance(msg.blocks[2], TextBlock)
        assert msg.content == 'BeforeAfter'
        assert msg.has_interleaving is True

    def test_roundtrip_simple(self):
        """Round-trip simple message through DB format"""
        original = Message(role='assistant', status='completed')
        original.blocks.append(TextBlock(text='Hello'))
        original.blocks.append(ActionBlock(id='a1', name='test', status='completed', body={'x': 1}))

        row = original.to_db_row()
        restored = Message.from_db_row(row)

        assert restored.content == original.content
        assert len(restored.actions) == len(original.actions)
        assert restored.actions[0].name == original.actions[0].name

    def test_roundtrip_interleaved(self):
        """Round-trip interleaved message through DB format"""
        original = Message(role='assistant', status='completed')
        original.blocks.append(TextBlock(text='Before'))
        original.blocks.append(ActionBlock(id='ws1', name='web_search', status='completed', body={'query': 'test'}, result={'results': [{'title': 'R1'}]}, server_executed=True))
        original.blocks.append(TextBlock(text='After'))

        row = original.to_db_row()
        restored = Message.from_db_row(row)

        assert len(restored.blocks) == len(original.blocks)
        assert [b.type for b in restored.blocks] == [b.type for b in original.blocks]
        assert restored.content == original.content
        assert restored.has_interleaving is True


class TestMessageLegacyConstruction:
    """Test backward-compatible construction"""

    def test_construct_with_content(self):
        """Construct message with content string"""
        msg = Message(role='user', content='Hello')

        assert msg.content == 'Hello'
        assert len(msg.blocks) == 1
        assert isinstance(msg.blocks[0], TextBlock)

    def test_construct_with_blocks(self):
        """Construct message with explicit blocks"""
        msg = Message(role='assistant', blocks=[
            TextBlock(text='Result'),
            ActionBlock(id='a1', name='test', status='completed', body={})
        ])

        assert msg.content == 'Result'
        assert len(msg.actions) == 1
        assert len(msg.blocks) == 2

    def test_construct_tool_message(self):
        """Tool message stores content separately"""
        msg = Message(role='tool', content='Tool result', action_id='a1')

        assert msg.content == 'Tool result'
        assert msg.action_id == 'a1'
        assert len(msg.blocks) == 0  # Tool content stored in _tool_content
