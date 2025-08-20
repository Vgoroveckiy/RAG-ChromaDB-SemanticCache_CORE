import json
from typing import Dict, List, Union
from langchain_core.documents import Document


class JsonTextSplitter:
    """Custom splitter for JSON files preserving structure"""

    def split_json(self, data: Union[Dict, List]) -> List[Document]:
        chunks = []
        if isinstance(data, list):
            # Process array items
            for item in data:
                chunks.append(
                    Document(
                        page_content=json.dumps(item, ensure_ascii=False),
                        metadata={"json_type": "array_item"},
                    )
                )
        elif isinstance(data, dict):
            # Process object properties
            for key, value in data.items():
                chunks.append(
                    Document(
                        page_content=json.dumps({key: value}, ensure_ascii=False),
                        metadata={"json_path": f"$.{key}"},
                    )
                )
        return chunks
