from dataclasses import dataclass
from typing import List
import json


@dataclass
class TextRegion:
    """Represents a text region detected by OCR."""
    x: int
    y: int
    width: int
    height: int
    text: str


@dataclass
class Frame:
    """Represents a video frame with OCR text regions."""
    image_filename: str
    image_width: int
    image_height: int
    text_regions: List[TextRegion]


@dataclass
class Program:
    """Represents a TV program with multiple frames."""
    name: str
    video_filename: str
    frames: List[Frame]


@dataclass
class OCRData:
    """Root data structure for OCR results."""
    source: str
    programs: List[Program]

    @classmethod
    def from_json(cls, json_path: str) -> 'OCRData':
        """Create OCRData instance from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        programs = []
        for program_data in data['programs']:
            frames = []
            for frame_data in program_data['frames']:
                text_regions = []
                for region_data in frame_data['text_regions']:
                    text_regions.append(TextRegion(**region_data))
                
                frames.append(Frame(
                    image_filename=frame_data['image_filename'],
                    image_width=frame_data['image_width'],
                    image_height=frame_data['image_height'],
                    text_regions=text_regions
                ))
            
            programs.append(Program(
                name=program_data['name'],
                video_filename=program_data['video_filename'],
                frames=frames
            ))
        
        return cls(
            source=data['source'],
            programs=programs
        )

    def to_json(self, json_path: str) -> None:
        """Save OCRData instance to JSON file."""
        data = {
            'source': self.source,
            'programs': [
                {
                    'name': program.name,
                    'video_filename': program.video_filename,
                    'frames': [
                        {
                            'image_filename': frame.image_filename,
                            'image_width': frame.image_width,
                            'image_height': frame.image_height,
                            'text_regions': [
                                {
                                    'x': region.x,
                                    'y': region.y,
                                    'width': region.width,
                                    'height': region.height,
                                    'text': region.text
                                }
                                for region in frame.text_regions
                            ]
                        }
                        for frame in program.frames
                    ]
                }
                for program in self.programs
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)