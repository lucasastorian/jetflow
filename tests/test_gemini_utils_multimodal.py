import base64

from google.genai import _common, models

from jetflow.clients.gemini.utils import messages_to_contents
from jetflow.models.message import ActionBlock, ImageBlock, Message, TextBlock


def _build_assistant_tool_messages(image_block: ImageBlock):
    assistant = Message(
        role="assistant",
        blocks=[ActionBlock(id="act_1", name="read_image", status="parsed", body={})],
    )

    tool = Message(
        role="tool",
        action_id="act_1",
        blocks=[TextBlock(text="OCR result"), image_block],
    )
    tool.content = "OCR result"
    return [assistant, tool]


def test_messages_to_contents_multimodal_tool_inline_data_has_matching_refs():
    img = ImageBlock(
        data=base64.b64encode(b"image-bytes").decode("ascii"),
        media_type="image/png",
    )
    contents = messages_to_contents(_build_assistant_tool_messages(img))

    tool_turn = contents[-1]
    assert isinstance(tool_turn, dict)
    assert tool_turn["role"] == "user"

    function_response = tool_turn["parts"][0]["function_response"]
    response = function_response["response"]
    multimodal_parts = function_response["parts"]

    assert "image_0" in response
    ref_name = response["image_0"]["$ref"]
    inline_data = multimodal_parts[0]["inlineData"]
    assert inline_data["displayName"] == ref_name
    assert inline_data["mimeType"] == "image/png"
    assert "inline_data" not in multimodal_parts[0]

    # Simulate SDK conversion stage to ensure camelCase survives.
    wire_payload = _common.convert_to_dict(models._Content_to_mldev(tool_turn))
    wire_inline = wire_payload["parts"][0]["functionResponse"]["parts"][0]["inlineData"]
    assert "displayName" in wire_inline
    assert "display_name" not in wire_inline


def test_messages_to_contents_multimodal_tool_url_uses_file_data():
    img = ImageBlock(
        url="https://example.com/image.png",
        media_type="image/png",
    )
    contents = messages_to_contents(_build_assistant_tool_messages(img))

    tool_turn = contents[-1]
    assert isinstance(tool_turn, dict)

    function_response = tool_turn["parts"][0]["function_response"]
    response = function_response["response"]
    multimodal_parts = function_response["parts"]
    file_data = multimodal_parts[0]["fileData"]

    assert file_data["fileUri"] == "https://example.com/image.png"
    assert file_data["mimeType"] == "image/png"
    assert file_data["displayName"] == response["image_0"]["$ref"]
