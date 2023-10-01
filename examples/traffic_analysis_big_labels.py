import argparse
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [
    np.array([[75, 300], [215, 300], [360, 400], [160, 420]]),
    np.array([[870, 360], [1110, 320], [1260, 440], [1050, 470]]),
    np.array([[900, 750], [1170, 600], [1475, 750], [1050, 860]]),
    np.array([[0, 460], [60, 440], [255, 610], [0, 620]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[370, 390], [245, 300], [410, 240], [600, 320]]),
    np.array([[1050, 420], [1260, 370], [1370, 455], [1165, 480]]),
    np.array([[700, 810], [930, 700], [1165, 850], [880, 935]]),
    np.array([[0, 430], [150, 420], [180, 540], [0, 550]]),
]


class DotAnnotator:
    """
    A class for drawing dot on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color: Union[sv.Color, sv.ColorPalette] = sv.ColorPalette.default(),
        thickness: int = 2,
        text_color: sv.Color = sv.Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):
        self.color: Union[sv.Color, sv.ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: sv.Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            python
            # >>> import supervision as sv
            #
            # >>> classes = ['person', ...]
            # >>> image = ...
            # >>> detections = sv.Detections(...)
            #
            # >>> box_annotator = sv.BoxAnnotator()
            # >>> labels = [
            # ...     f"{classes[class_id]} {confidence:0.2f}"
            # ...     for _, _, confidence, class_id, _
            # ...     in detections
            # ... ]
            # >>> annotated_frame = box_annotator.annotate(
            # ...     scene=image.copy(),
            # ...     detections=detections,
            # ...     labels=labels
            # ... )

        """
        font = cv2.FONT_HERSHEY_DUPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            x1 = int((x1 + x2) / 2)
            y1 = int((y1 + y2) / 2)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, sv.ColorPalette)
                else self.color
            )

            cv2.circle(
                img=scene,
                center=(x1, y1),
                radius=4,
                color=color.as_bgr(),
                thickness=4
            )
            # cv2.rectangle(
            #     img=scene,
            #     pt1=(x1, y1),
            #     pt2=(x2, y2),
            #     color=color.as_bgr(),
            #     thickness=self.thickness,
            # )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            y1 = y1 - 20   # label shifting up from center dot

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
                print(f'ID по зонам: {self.tracker_id_to_zone_id}')

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)

        # TODO: ГДЕ-ТО НУЖНО ДОБАВИТЬ ФИЛЬТРАЦИЮ КЛАССОВ
        # detections_all.class_id = np.vectorize(
        #     lambda x: self.filter(...)
        # filtering out detections with unwanted classes
        # detections_all.filter(mask=mask, inplace=True)

        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.BOTTOM_CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.BOTTOM_CENTER
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.BOTTOM_CENTER
        )

        # self.box_annotator = sv.BoxAnnotator(color=COLORS)
        # self.box_annotator = sv.MaskAnnotator(color=COLORS)

        # Label, font size
        self.box_annotator = DotAnnotator(
            color=COLORS,
            text_scale=7,
            text_padding=30,
            text_thickness=10
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.BOTTOM_CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                # for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                for frame in frame_generator:
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            # for frame in tqdm(frame_generator, total=self.video_info.total_frames):
            for frame in frame_generator:
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"ID{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        text_scale=2,
                        text_thickness=6,
                        text_padding=20,
                        background_color=COLORS.colors[zone_in_id],
                    )
                    # print(count)
                    # print(detections)
                print(counts)

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # TODO: Сделать список классов аргументом командной строки
        CLASS_ID = np.array([1, 2, 3, 4, 5, 6, 7]).astype(int)

        # Маска для нежелательных классов
        mask = [True if class_id in CLASS_ID else False for class_id in detections.class_id]
        detections = detections[mask]
        # print(detections)
        # Конец фильтра


        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

            # []
            print(f'Въезд: {i} - {detections_in_zones}')
            print(f'Выезд: {i} - {detections_out_zone}')

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
