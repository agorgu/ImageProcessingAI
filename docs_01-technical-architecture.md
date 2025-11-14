# Teknik Mimari Dökümanı / Technical Architecture Documentation

## 🏗️ Sistem Mimarisi Genel Bakış / System Architecture Overview

### TR: Genel Yapı

ImageProcessingAI platformu, 5 ana katmandan oluşan bir mimari üzerine inşa edilmiştir:

1. **Kamera Katmanı (Edge Layer)** - Video akışı yakalama
2. **Video İşleme Katmanı (Processing Layer)** - Görüntü ön işleme
3. **AI/ML Katmanı (AI Layer)** - Nesne tespiti ve takip
4. **Backend Katmanı (API Layer)** - İş mantığı ve veri yönetimi
5. **Frontend Katmanı (UI Layer)** - Kullanıcı arayüzü ve dashboard

---

### EN: General Structure

ImageProcessingAI platform is built on a 5-layer architecture:

1. **Edge Layer** - Video stream capture
2. **Processing Layer** - Image preprocessing
3. **AI Layer** - Object detection and tracking
4. **API Layer** - Business logic and data management
5. **UI Layer** - User interface and dashboard

---

## 🎥 1. KAMERA KATMANI / CAMERA LAYER

### TR: Kamera Entegrasyonu

#### Desteklenen Kamera Tipleri:
- **IP Kameralar**: RTSP, ONVIF protokolleri
- **Analog Kameralar**: DVR/NVR üzerinden dijital dönüşüm
- **USB Kameralar**: Lokal test ve küçük işletmeler için
- **Mobil Kameralar**: Tablet/telefon entegrasyonu (gelecek)

#### Teknik Özellikler:
```
- Çözünürlük: Minimum 720p (1280x720), Önerilen 1080p (1920x1080)
- FPS: Minimum 15 FPS, Önerilen 25-30 FPS
- Codec: H.264, H.265 (HEVC)
- Protokol: RTSP (rtsp://), HTTP/HTTPS
- Latency: <500ms (gerçek zamanlı analiz için)
```

#### Kamera Bağlantı Süreci:

**Adım 1: Kamera Keşfi (Camera Discovery)**
```python
# ONVIF protokolü ile ağdaki kameraları otomatik bul
from onvif import ONVIFCamera

def discover_cameras(network_range):
    """
    Ağdaki ONVIF uyumlu kameraları keşfet
    """
    cameras = []
    for ip in network_range:
        try:
            camera = ONVIFCamera(ip, 80, 'admin', 'password')
            camera_info = camera.devicemgmt.GetDeviceInformation()
            cameras.append({
                'ip': ip,
                'manufacturer': camera_info.Manufacturer,
                'model': camera_info.Model,
                'serial': camera_info.SerialNumber
            })
        except:
            continue
    return cameras
```

**Adım 2: RTSP Stream Bağlantısı**
```python
import cv2

def connect_to_camera(rtsp_url):
    """
    RTSP URL'den video stream'i al
    rtsp_url: "rtsp://username:password@ip:port/stream"
    """
    cap = cv2.VideoCapture(rtsp_url)
    
    # Bağlantı ayarları
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Düşük latency için
    cap.set(cv2.CAP_PROP_FPS, 25)
    
    return cap
```

**Adım 3: Frame Yakalama ve Buffer Yönetimi**
```python
def capture_frames(camera_stream):
    """
    Kameradan frame'leri sürekli yakala
    """
    while True:
        ret, frame = camera_stream.read()
        
        if not ret:
            # Bağlantı koptu, yeniden bağlan
            reconnect_camera()
            continue
        
        # Frame'i işleme kuyruğuna gönder
        frame_queue.put({
            'timestamp': time.time(),
            'camera_id': camera_id,
            'frame': frame
        })
```

#### Bant Genişliği Optimizasyonu:
```
Tek kamera:
- 1080p @ 25 FPS + H.264 = ~2-4 Mbps
- 720p @ 25 FPS + H.265 = ~1-2 Mbps

10 kamera sistemi:
- Toplam bandwidth: 20-40 Mbps
- Önerilen internet: Minimum 50 Mbps upload
```

---

### EN: Camera Integration

#### Supported Camera Types:
- **IP Cameras**: RTSP, ONVIF protocols
- **Analog Cameras**: Digital conversion via DVR/NVR
- **USB Cameras**: For local testing and small businesses
- **Mobile Cameras**: Tablet/phone integration (future)

#### Technical Specifications:
```
- Resolution: Minimum 720p (1280x720), Recommended 1080p (1920x1080)
- FPS: Minimum 15 FPS, Recommended 25-30 FPS
- Codec: H.264, H.265 (HEVC)
- Protocol: RTSP (rtsp://), HTTP/HTTPS
- Latency: <500ms (for real-time analysis)
```

#### Camera Connection Process:

**Step 1: Camera Discovery**
```python
from onvif import ONVIFCamera

def discover_cameras(network_range):
    """
    Discover ONVIF-compatible cameras on network
    """
    cameras = []
    for ip in network_range:
        try:
            camera = ONVIFCamera(ip, 80, 'admin', 'password')
            camera_info = camera.devicemgmt.GetDeviceInformation()
            cameras.append({
                'ip': ip,
                'manufacturer': camera_info.Manufacturer,
                'model': camera_info.Model,
                'serial': camera_info.SerialNumber
            })
        except:
            continue
    return cameras
```

**Step 2: RTSP Stream Connection**
```python
import cv2

def connect_to_camera(rtsp_url):
    """
    Get video stream from RTSP URL
    rtsp_url: "rtsp://username:password@ip:port/stream"
    """
    cap = cv2.VideoCapture(rtsp_url)
    
    # Connection settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
    cap.set(cv2.CAP_PROP_FPS, 25)
    
    return cap
```

**Step 3: Frame Capture and Buffer Management**
```python
def capture_frames(camera_stream):
    """
    Continuously capture frames from camera
    """
    while True:
        ret, frame = camera_stream.read()
        
        if not ret:
            # Connection lost, reconnect
            reconnect_camera()
            continue
        
        # Send frame to processing queue
        frame_queue.put({
            'timestamp': time.time(),
            'camera_id': camera_id,
            'frame': frame
        })
```

---

## 🖼️ 2. VIDEO İŞLEME KATMANI / VIDEO PROCESSING LAYER

### TR: Görüntü Ön İşleme

Kameradan gelen raw frame'ler AI modeline girmeden önce optimize edilir.

#### Adım 1: Frame Preprocessing
```python
import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Frame'i AI modeli için hazırla
    """
    # 1. Resize (YOLOv8 için 640x640)
    resized = cv2.resize(frame, (640, 640))
    
    # 2. Normalizasyon (0-255 -> 0-1)
    normalized = resized / 255.0
    
    # 3. RGB -> BGR dönüşümü (OpenCV BGR kullanır)
    rgb_frame = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    # 4. Tensor formatına çevir (Batch, Channel, Height, Width)
    tensor = np.transpose(rgb_frame, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor
```

#### Adım 2: Görüntü İyileştirme (Enhancement)
```python
def enhance_image(frame):
    """
    Düşük ışık ve gürültülü ortamlar için görüntü kalitesini artır
    """
    # Histogram Equalization (kontrast artırma)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Noise reduction (gürültü azaltma)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return denoised
```

#### Adım 3: ROI (Region of Interest) Belirleme
```python
def define_roi(frame, roi_config):
    """
    İlgilenilen bölgeleri işaretle (örn: kasa alanı, masa bölgeleri)
    """
    height, width = frame.shape[:2]
    
    # Kasa alanı (POS area)
    pos_area = {
        'x1': int(width * 0.7),  # Sağ üst köşe
        'y1': int(height * 0.1),
        'x2': int(width * 0.95),
        'y2': int(height * 0.3),
        'name': 'POS_AREA'
    }
    
    # Masa bölgeleri
    table_areas = []
    for table in roi_config['tables']:
        table_areas.append({
            'x1': table['x1'],
            'y1': table['y1'],
            'x2': table['x2'],
            'y2': table['y2'],
            'table_number': table['number']
        })
    
    return {
        'pos_area': pos_area,
        'table_areas': table_areas
    }
```

---

### EN: Image Preprocessing

Raw frames from cameras are optimized before entering the AI model.

#### Step 1: Frame Preprocessing
```python
import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Prepare frame for AI model
    """
    # 1. Resize (640x640 for YOLOv8)
    resized = cv2.resize(frame, (640, 640))
    
    # 2. Normalization (0-255 -> 0-1)
    normalized = resized / 255.0
    
    # 3. RGB -> BGR conversion (OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    # 4. Convert to tensor format (Batch, Channel, Height, Width)
    tensor = np.transpose(rgb_frame, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor
```

---

## 🤖 3. AI/ML KATMANI / AI/ML LAYER

### TR: Yapay Zeka Pipeline

Bu katman 3 ana AI modülünden oluşur:

#### Modül 1: Nesne Tespiti (Object Detection) - YOLOv8

**Model Özellikleri:**
```
Model: YOLOv8n (nano) veya YOLOv8s (small)
Input: 640x640 RGB image
Output: Bounding boxes + class + confidence
Classes: 
  - person (insan)
  - cash (nakit para - custom trained)
  - hand (el - custom trained)
  - pos_terminal (POS cihazı - custom trained)
Inference Time: 
  - CPU: ~50-80ms per frame
  - GPU (T4): ~5-10ms per frame
  - GPU (A100): ~2-5ms per frame
```

**Kod İmplementasyonu:**
```python
from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        YOLOv8 modelini yükle
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Custom trained weights'i yükle (para tespiti için)
        self.model.load('custom_cash_detection.pt')
    
    def detect(self, frame):
        """
        Frame içindeki nesneleri tespit et
        """
        results = self.model(frame, conf=0.5)  # Confidence threshold: 0.5
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'center': self._calculate_center(box.xyxy[0])
                }
                detections.append(detection)
        
        return detections
    
    def _calculate_center(self, bbox):
        """Bounding box'ın merkez noktasını hesapla"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (float(center_x), float(center_y))
```

**Para Tespiti için Custom Model Training:**
```python
from ultralytics import YOLO

def train_custom_cash_detection():
    """
    Para tespiti için custom YOLOv8 modeli eğit
    """
    # Base model
    model = YOLO('yolov8n.pt')
    
    # Dataset yapısı:
    # dataset/
    #   ├── images/
    #   │   ├── train/
    #   │   └── val/
    #   └── labels/
    #       ├── train/
    #       └── val/
    
    # Training parametreleri
    results = model.train(
        data='cash_dataset.yaml',  # Dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',
        project='cash_detection',
        name='yolov8n_cash',
        
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10,   # Rotation
        translate=0.1,  # Translation
        scale=0.5,    # Scale
        flipud=0.0,   # Flip up-down
        fliplr=0.5,   # Flip left-right
        mosaic=1.0,   # Mosaic augmentation
    )
    
    return model

# Dataset YAML örneği (cash_dataset.yaml):
"""
path: /path/to/dataset
train: images/train
val: images/val

nc: 4  # number of classes
names: ['person', 'cash', 'hand', 'pos_terminal']
"""
```

**Para Sayma Algoritması:**
```python
class CashCounter:
    def __init__(self):
        # Türk Lirası banknot boyutları (mm)
        self.banknote_sizes = {
            '5': (120, 64),
            '10': (126, 64),
            '20': (132, 64),
            '50': (138, 64),
            '100': (144, 64),
            '200': (150, 64)
        }
        
        # Renk profilleri (HSV color space)
        self.banknote_colors = {
            '5': {'h': [160, 180], 's': [50, 255], 'v': [50, 255]},  # Gri-mor
            '10': {'h': [0, 15], 's': [100, 255], 'v': [100, 255]},  # Kırmızı
            '20': {'h': [90, 110], 's': [50, 255], 'v': [50, 255]},  # Yeşil
            '50': {'h': [15, 30], 's': [100, 255], 'v': [150, 255]}, # Turuncu
            '100': {'h': [0, 180], 's': [0, 50], 'v': [150, 255]},   # Mavi-gri
            '200': {'h': [20, 40], 's': [100, 255], 'v': [100, 255]} # Sarı
        }
    
    def count_cash(self, detections, frame):
        """
        Tespit edilen paraları say ve topla
        """
        total_amount = 0
        cash_items = []
        
        for det in detections:
            if det['class'] == 'cash':
                # Banknotu kırp
                x1, y1, x2, y2 = det['bbox']
                banknote_img = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Banknot değerini tespit et
                denomination = self._identify_denomination(banknote_img)
                
                if denomination:
                    total_amount += int(denomination)
                    cash_items.append({
                        'denomination': denomination,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
        
        return {
            'total_amount': total_amount,
            'items': cash_items,
            'count': len(cash_items)
        }
    
    def _identify_denomination(self, banknote_img):
        """
        Banknot görselinden değeri tespit et (renk + boyut analizi)
        """
        # HSV color space'e çevir
        hsv = cv2.cvtColor(banknote_img, cv2.COLOR_BGR2HSV)
        
        # Her banknot için renk eşleşmesi kontrol et
        best_match = None
        best_score = 0
        
        for denom, color_range in self.banknote_colors.items():
            # Color mask oluştur
            lower = np.array([color_range['h'][0], color_range['s'][0], color_range['v'][0]])
            upper = np.array([color_range['h'][1], color_range['s'][1], color_range['v'][1]])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Eşleşme skorunu hesapla
            score = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            
            if score > best_score:
                best_score = score
                best_match = denom
        
        # Minimum threshold kontrolü
        if best_score > 0.3:  # %30'dan fazla eşleşme
            return best_match
        else:
            return None
```

---

#### Modül 2: Kişi Takibi (Person Tracking) - DeepSORT

**Model Özellikleri:**
```
Algorithm: DeepSORT (Deep Learning + SORT)
Purpose: Kişileri frame'ler arasında takip et (garsonları ID'lendir)
Input: Bounding boxes from YOLOv8
Output: Unique tracking ID per person
Features:
  - Re-identification (yeniden tanıma)
  - Occlusion handling (kapatılma durumları)
  - Track lifecycle management
```

**Kod İmplementasyonu:**
```python
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self):
        """
        DeepSORT tracker'ı başlat
        """
        self.tracker = DeepSort(
            max_age=30,  # Track'i 30 frame boyunca sakla
            n_init=3,    # 3 frame'de görünce ID ver
            max_iou_distance=0.7,  # IOU threshold
            embedder="mobilenet",  # Re-ID model
            half=True,  # FP16 precision (GPU için)
            embedder_gpu=True
        )
        
        self.tracks = {}  # Track history
    
    def update(self, detections, frame):
        """
        Tespit edilen kişileri track'le
        """
        # Sadece 'person' class'ını filtrele
        person_detections = []
        for det in detections:
            if det['class'] == 'person':
                # DeepSORT formatı: ([x1, y1, x2, y2], confidence, class)
                person_detections.append((
                    det['bbox'],
                    det['confidence'],
                    'person'
                ))
        
        # Tracking güncelle
        tracks = self.tracker.update_tracks(person_detections, frame=frame)
        
        # Track bilgilerini kaydet
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            
            # Track history'yi güncelle
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'id': track_id,
                    'first_seen': time.time(),
                    'positions': [],
                    'events': []
                }
            
            self.tracks[track_id]['positions'].append({
                'timestamp': time.time(),
                'bbox': bbox,
                'center': self._calculate_center(bbox)
            })
            
            active_tracks.append({
                'track_id': track_id,
                'bbox': bbox,
                'center': self._calculate_center(bbox)
            })
        
        return active_tracks
    
    def _calculate_center(self, bbox):
        """Track'in merkez noktası"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_trajectory(self, track_id, last_n_seconds=10):
        """
        Bir kişinin son N saniyedeki hareketini al
        """
        if track_id not in self.tracks:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - last_n_seconds
        
        trajectory = []
        for pos in self.tracks[track_id]['positions']:
            if pos['timestamp'] >= cutoff_time:
                trajectory.append(pos['center'])
        
        return trajectory
```

---

#### Modül 3: Olay Tespiti (Event Detection)

**Garson-Para-Kasa Korelasyon Algoritması:**

```python
class EventDetector:
    def __init__(self, pos_area_config):
        """
        Olay tespit motoru
        """
        self.pos_area = pos_area_config  # Kasa alanı koordinatları
        self.active_events = {}  # Devam eden olaylar
        self.completed_events = []  # Tamamlanan olaylar
        
        # Thresholds
        self.CASH_PICKUP_DISTANCE = 100  # pixels
        self.POS_VISIT_TIME_LIMIT = 120  # seconds (2 dakika)
        self.HAND_CASH_IOU_THRESHOLD = 0.3  # Intersection over Union
    
    def process_frame(self, detections, tracks, frame_id, timestamp):
        """
        Her frame için olayları analiz et
        """
        # 1. Para tespitleri
        cash_detections = [d for d in detections if d['class'] == 'cash']
        
        # 2. El tespitleri
        hand_detections = [d for d in detections if d['class'] == 'hand']
        
        # 3. Kişi track'leri
        persons = tracks
        
        # 4. Para alma olayını tespit et
        for cash in cash_detections:
            for hand in hand_detections:
                # El ve para yakın mı?
                if self._is_close(cash['center'], hand['center'], self.CASH_PICKUP_DISTANCE):
                    # Bu ele ait kişiyi bul
                    person = self._find_person_by_hand(hand, persons)
                    
                    if person:
                        self._register_cash_pickup_event(
                            person['track_id'],
                            cash,
                            timestamp,
                            frame_id
                        )
        
        # 5. Kasa ziyaretini kontrol et
        for person in persons:
            if self._is_in_pos_area(person['center']):
                self._register_pos_visit(person['track_id'], timestamp)
        
        # 6. Alert kontrolü - Para aldı ama kasaya gitmedi mi?
        self._check_for_alerts(timestamp)
        
        return self.active_events
    
    def _register_cash_pickup_event(self, track_id, cash_detection, timestamp, frame_id):
        """
        Para alma olayını kaydet
        """
        event_id = f"cash_pickup_{track_id}_{timestamp}"
        
        self.active_events[event_id] = {
            'event_id': event_id,
            'type': 'CASH_PICKUP',
            'waiter_id': track_id,
            'cash_amount': cash_detection.get('denomination', 0),
            'pickup_time': timestamp,
            'pickup_frame': frame_id,
            'pos_visited': False,
            'pos_visit_time': None,
            'status': 'PENDING',  # PENDING, COMPLETED, ALERT
            'bbox': cash_detection['bbox']
        }
    
    def _register_pos_visit(self, track_id, timestamp):
        """
        Kasa ziyaretini kaydet
        """
        # Bu kişinin bekleyen para alma olayı var mı?
        for event_id, event in self.active_events.items():
            if (event['waiter_id'] == track_id and 
                event['type'] == 'CASH_PICKUP' and 
                not event['pos_visited']):
                
                # Olayı güncelle
                event['pos_visited'] = True
                event['pos_visit_time'] = timestamp
                event['status'] = 'COMPLETED'
                
                # Tamamlanmış olaylara taşı
                self.completed_events.append(event)
                del self.active_events[event_id]
                
                break
    
    def _check_for_alerts(self, current_timestamp):
        """
        Alert oluşturulması gereken durumları kontrol et
        """
        alerts = []
        
        for event_id, event in list(self.active_events.items()):
            if event['type'] == 'CASH_PICKUP' and event['status'] == 'PENDING':
                # Para alındıktan sonra geçen süre
                elapsed_time = current_timestamp - event['pickup_time']
                
                # 2 dakika içinde kasaya gitmedi mi?
                if elapsed_time > self.POS_VISIT_TIME_LIMIT:
                    event['status'] = 'ALERT'
                    
                    alert = {
                        'alert_id': f"alert_{event_id}",
                        'type': 'CASH_NOT_DEPOSITED',
                        'severity': 'HIGH',
                        'waiter_id': event['waiter_id'],
                        'cash_amount': event['cash_amount'],
                        'elapsed_time': elapsed_time,
                        'timestamp': current_timestamp,
                        'message': f"Waiter {event['waiter_id']} picked up {event['cash_amount']} TL "
                                   f"but did not visit POS for {int(elapsed_time)}s"
                    }
                    
                    alerts.append(alert)
                    
                    # Alert'i kaydet ve active events'ten çıkar
                    self.completed_events.append(event)
                    del self.active_events[event_id]
        
        return alerts
    
    def _is_close(self, point1, point2, threshold):
        """İki nokta birbirine yakın mı?"""
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        return distance < threshold
    
    def _is_in_pos_area(self, point):
        """Nokta kasa alanında mı?"""
        x, y = point
        return (self.pos_area['x1'] <= x <= self.pos_area['x2'] and
                self.pos_area['y1'] <= y <= self.pos_area['y2'])
    
    def _find_person_by_hand(self, hand_detection, persons):
        """El tespitten kişiyi bul (en yakın person bbox)"""
        hand_center = hand_detection['center']
        
        min_distance = float('inf')
        closest_person = None
        
        for person in persons:
            person_center = person['center']
            distance = np.sqrt((hand_center[0] - person_center[0])**2 + 
                               (hand_center[1] - person_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_person = person
        
        return closest_person if min_distance < 200 else None  # Max 200 pixel
```

---

### EN: AI Pipeline

This layer consists of 3 main AI modules:

#### Module 1: Object Detection - YOLOv8

**Model Specifications:**
```
Model: YOLOv8n (nano) or YOLOv8s (small)
Input: 640x640 RGB image
Output: Bounding boxes + class + confidence
Classes: 
  - person
  - cash (custom trained)
  - hand (custom trained)
  - pos_terminal (custom trained)
Inference Time: 
  - CPU: ~50-80ms per frame
  - GPU (T4): ~5-10ms per frame
  - GPU (A100): ~2-5ms per frame
```

#### Module 2: Person Tracking - DeepSORT

Tracks individuals across frames with unique IDs, enabling trajectory analysis and behavior monitoring.

#### Module 3: Event Detection

Correlates cash pickups, waiter movements, and POS visits to detect anomalies and trigger alerts.

---

## 🔗 4. BACKEND KATMANI / BACKEND LAYER

### TR: API ve İş Mantığı

Backend, FastAPI framework'ü üzerine kurulu mikroservis mimarisi kullanır.

#### Servis Mimarisi:

```
Backend Services:
├── Video Ingestion Service (Video alma)
├── AI Processing Service (AI işleme)
├── Event Management Service (Olay yönetimi)
├── Alert Service (Bildirim servisi)
├── Analytics Service (Analitik)
└── API Gateway (Ana API)
```

#### Ana API Endpoints:

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="ImageProcessingAI API", version="1.0.0")

# 1. Kamera Yönetimi
@app.post("/api/v1/cameras/register")
async def register_camera(camera_config: CameraConfig):
    """
    Yeni kamera kaydet
    
    Request Body:
    {
        "name": "Restoran Ana Salon Kamera 1",
        "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream",
        "location": {
            "restaurant_id": "rest_001",
            "area": "main_dining"
        },
        "roi_config": {
            "pos_area": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
            "tables": [...]
        }
    }
    
    Response:
    {
        "camera_id": "cam_uuid_123",
        "status": "registered",
        "stream_status": "active"
    }
    """
    camera_id = camera_service.register(camera_config)
    
    # Video stream'i başlat
    await video_ingestion_service.start_stream(camera_id)
    
    return {"camera_id": camera_id, "status": "registered"}


# 2. Real-time Video Stream (WebSocket)
@app.websocket("/ws/camera/{camera_id}/stream")
async def camera_stream(websocket: WebSocket, camera_id: str):
    """
    Kameradan real-time video stream
    WebSocket üzerinden JPEG frame'ler gönder
    """
    await websocket.accept()
    
    try:
        while True:
            # Frame al
            frame = await video_service.get_latest_frame(camera_id)
            
            # AI sonuçlarını overlay et
            annotated_frame = await ai_service.annotate_frame(frame, camera_id)
            
            # JPEG'e çevir
            _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # WebSocket'e gönder
            await websocket.send_bytes(buffer.tobytes())
            
            await asyncio.sleep(0.04)  # ~25 FPS
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# 3. Olayları Getir
@app.get("/api/v1/events")
async def get_events(
    restaurant_id: str,
    start_date: datetime,
    end_date: datetime,
    event_type: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Belirli tarih aralığındaki olayları getir
    
    Query Parameters:
    - restaurant_id: Restoran ID
    - start_date: Başlangıç tarihi (ISO 8601)
    - end_date: Bitiş tarihi
    - event_type: CASH_PICKUP, POS_VISIT, etc.
    - status: PENDING, COMPLETED, ALERT
    
    Response:
    {
        "total": 150,
        "events": [
            {
                "event_id": "evt_123",
                "type": "CASH_PICKUP",
                "waiter_id": 5,
                "cash_amount": 250,
                "timestamp": "2025-01-14T16:30:00Z",
                "status": "ALERT",
                "camera_id": "cam_001",
                "video_clip_url": "https://cdn.../clip_123.mp4"
            },
            ...
        ]
    }
    """
    events = await event_service.query_events(
        restaurant_id=restaurant_id,
        start_date=start_date,
        end_date=end_date,
        event_type=event_type,
        status=status
    )
    
    return {"total": len(events), "events": events}


# 4. Alert Bildirimleri (WebSocket)
@app.websocket("/ws/alerts/{restaurant_id}")
async def alert_stream(websocket: WebSocket, restaurant_id: str):
    """
    Real-time alert bildirimleri
    """
    await websocket.accept()
    
    # Alert kanalına abone ol
    channel = f"alerts:{restaurant_id}"
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                alert_data = json.loads(message['data'])
                await websocket.send_json(alert_data)
    
    except Exception as e:
        print(f"Alert stream error: {e}")
    finally:
        pubsub.unsubscribe(channel)
        await websocket.close()


# 5. Video Clip İndirme
@app.get("/api/v1/events/{event_id}/video")
async def get_event_video(event_id: str):
    """
    Bir olaya ait video clip'ini indir
    
    Olay öncesi 10 saniye + olay süresi + olay sonrası 10 saniye
    """
    event = await event_service.get_event(event_id)
    
    # Video clip'i oluştur
    video_clip = await video_service.create_clip(
        camera_id=event['camera_id'],
        start_time=event['timestamp'] - timedelta(seconds=10),
        end_time=event['timestamp'] + timedelta(seconds=30)
    )
    
    return StreamingResponse(
        video_clip,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"attachment; filename=event_{event_id}.mp4"
        }
    )


# 6. Analitik - Günlük Özet
@app.get("/api/v1/analytics/daily-summary")
async def daily_summary(restaurant_id: str, date: date):
    """
    Günlük özet istatistikler
    
    Response:
    {
        "date": "2025-01-14",
        "restaurant_id": "rest_001",
        "statistics": {
            "total_cash_handled": 15250.50,
            "total_transactions": 127,
            "alerts_generated": 3,
            "average_cash_per_transaction": 120.08,
            "busiest_hour": "19:00-20:00",
            "waiter_stats": [
                {
                    "waiter_id": 5,
                    "cash_handled": 3500,
                    "transactions": 28,
                    "alerts": 1,
                    "pos_compliance_rate": 0.96
                },
                ...
            ]
        }
    }
    """
    summary = await analytics_service.generate_daily_summary(
        restaurant_id=restaurant_id,
        date=date
    )
    
    return summary


# 7. POS Entegrasyonu
@app.post("/api/v1/pos/transaction")
async def pos_transaction_webhook(transaction: POSTransaction):
    """
    POS sisteminden gelen transaction webhook'u
    
    AI tespit edilen nakit ödemeleri ile karşılaştır
    
    Request Body:
    {
        "transaction_id": "pos_txn_123",
        "timestamp": "2025-01-14T16:30:00Z",
        "amount": 250.00,
        "payment_method": "cash",
        "waiter_id": 5,
        "table_number": 12
    }
    """
    # AI event'leri ile karşılaştır
    matching_event = await event_service.find_matching_ai_event(
        timestamp=transaction.timestamp,
        amount=transaction.amount,
        waiter_id=transaction.waiter_id,
        tolerance_seconds=30,
        tolerance_amount=10  # +/- 10 TL tolerance
    )
    
    if matching_event:
        # Eşleşme var - OK
        await event_service.mark_as_verified(matching_event['event_id'])
        status = "VERIFIED"
    else:
        # Eşleşme yok - Potansiyel sorun
        await alert_service.create_alert({
            "type": "POS_AI_MISMATCH",
            "severity": "MEDIUM",
            "pos_transaction": transaction,
            "message": "POS transaction has no matching AI detection"
        })
        status = "UNVERIFIED"
    
    return {"status": status}
```

#### Veritabanı Şeması:

```sql
-- Restaurants (Restoranlar)
CREATE TABLE restaurants (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    address TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cameras (Kameralar)
CREATE TABLE cameras (
    id UUID PRIMARY KEY,
    restaurant_id UUID REFERENCES restaurants(id),
    name VARCHAR(255),
    rtsp_url TEXT,
    status VARCHAR(50), -- active, inactive, error
    roi_config JSONB,  -- ROI configuration
    created_at TIMESTAMP DEFAULT NOW()
);

-- Events (Olaylar)
CREATE TABLE events (
    id UUID PRIMARY KEY,
    camera_id UUID REFERENCES cameras(id),
    event_type VARCHAR(50), -- CASH_PICKUP, POS_VISIT, etc.
    waiter_track_id INTEGER,
    cash_amount DECIMAL(10, 2),
    pickup_timestamp TIMESTAMP,
    pos_visit_timestamp TIMESTAMP,
    status VARCHAR(50), -- PENDING, COMPLETED, ALERT
    metadata JSONB,  -- Ekstra bilgiler
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_camera_timestamp (camera_id, pickup_timestamp),
    INDEX idx_status (status),
    INDEX idx_waiter (waiter_track_id)
);

-- Alerts (Alarmlar)
CREATE TABLE alerts (
    id UUID PRIMARY KEY,
    event_id UUID REFERENCES events(id),
    alert_type VARCHAR(50),
    severity VARCHAR(20), -- LOW, MEDIUM, HIGH, CRITICAL
    message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID,
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_severity (severity),
    INDEX idx_acknowledged (acknowledged)
);

-- Video Clips (Video Kayıtları)
CREATE TABLE video_clips (
    id UUID PRIMARY KEY,
    event_id UUID REFERENCES events(id),
    camera_id UUID REFERENCES cameras(id),
    start_timestamp TIMESTAMP,
    end_timestamp TIMESTAMP,
    duration_seconds INTEGER,
    file_path TEXT,
    file_size_bytes BIGINT,
    thumbnail_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analytics Cache (Analitik Önbellek)
CREATE TABLE analytics_cache (
    id UUID PRIMARY KEY,
    restaurant_id UUID REFERENCES restaurants(id),
    date DATE,
    metric_type VARCHAR(100), -- daily_summary, waiter_performance, etc.
    data JSONB,
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(restaurant_id, date, metric_type)
);
```

---

### EN: API and Business Logic

Backend uses microservice architecture built on FastAPI framework.

[Similar structure in English - abbreviated for space]

---

## 🖥️ 5. FRONTEND KATMANI / FRONTEND LAYER

### TR: Kullanıcı Arayüzü

React tabanlı modern, responsive dashboard.

#### Ana Ekranlar:

**1. Live Monitoring (Canlı İzleme)**
```jsx
import React, { useEffect, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';

function LiveMonitoring({ restaurantId }) {
    const [cameras, setCameras] = useState([]);
    const [alerts, setAlerts] = useState([]);
    
    // WebSocket bağlantısı - Alert'ler için
    const { messages: alertMessages } = useWebSocket(
        `ws://api.example.com/ws/alerts/${restaurantId}`
    );
    
    useEffect(() => {
        // Yeni alert geldiğinde
        if (alertMessages.length > 0) {
            const latestAlert = alertMessages[alertMessages.length - 1];
            setAlerts(prev => [latestAlert, ...prev]);
            
            // Ses bildirimi
            playAlertSound();
            
            // Push notification
            showNotification(latestAlert);
        }
    }, [alertMessages]);
    
    return (
        <div className="monitoring-grid">
            {/* Kamera Grid */}
            <div className="camera-grid">
                {cameras.map(camera => (
                    <CameraView 
                        key={camera.id} 
                        cameraId={camera.id}
                        showOverlay={true}
                    />
                ))}
            </div>
            
            {/* Alert Panel */}
            <div className="alert-panel">
                <h2>Live Alerts</h2>
                {alerts.map(alert => (
                    <AlertCard key={alert.id} alert={alert} />
                ))}
            </div>
        </div>
    );
}

// Tek kamera görünümü
function CameraView({ cameraId, showOverlay }) {
    const canvasRef = useRef(null);
    const ws = useRef(null);
    
    useEffect(() => {
        // WebSocket video stream
        ws.current = new WebSocket(
            `ws://api.example.com/ws/camera/${cameraId}/stream`
        );
        
        ws.current.onmessage = (event) => {
            const blob = event.data;
            const url = URL.createObjectURL(blob);
            
            const img = new Image();
            img.onload = () => {
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
            };
            img.src = url;
        };
        
        return () => {
            ws.current.close();
        };
    }, [cameraId]);
    
    return (
        <div className="camera-view">
            <canvas ref={canvasRef} width={640} height={480} />
            <div className="camera-info">
                <span>Camera {cameraId}</span>
                <span className="status-indicator active">●</span>
            </div>
        </div>
    );
}
```

**2. Events Timeline (Olay Zaman Çizelgesi)**
```jsx
function EventsTimeline({ restaurantId, date }) {
    const [events, setEvents] = useState([]);
    
    useEffect(() => {
        // API'den olayları getir
        fetch(`/api/v1/events?restaurant_id=${restaurantId}&date=${date}`)
            .then(res => res.json())
            .then(data => setEvents(data.events));
    }, [restaurantId, date]);
    
    return (
        <div className="timeline">
            {events.map(event => (
                <TimelineEvent 
                    key={event.id}
                    event={event}
                    onClick={() => showEventDetail(event)}
                />
            ))}
        </div>
    );
}

function TimelineEvent({ event, onClick }) {
    const statusColor = {
        'COMPLETED': 'green',
        'PENDING': 'yellow',
        'ALERT': 'red'
    }[event.status];
    
    return (
        <div className="timeline-event" onClick={onClick}>
            <div className="event-time">
                {new Date(event.timestamp).toLocaleTimeString()}
            </div>
            <div className="event-content">
                <span className={`status-badge ${statusColor}`}>
                    {event.status}
                </span>
                <span className="event-description">
                    Waiter #{event.waiter_id} - {event.cash_amount} TL
                </span>
            </div>
            <button className="play-video-btn">
                <PlayIcon /> Watch Video
            </button>
        </div>
    );
}
```

**3. Analytics Dashboard**
```jsx
import { LineChart, BarChart, PieChart } from 'recharts';

function AnalyticsDashboard({ restaurantId, dateRange }) {
    const [analytics, setAnalytics] = useState(null);
    
    useEffect(() => {
        fetch(`/api/v1/analytics/summary?restaurant_id=${restaurantId}&date_range=${dateRange}`)
            .then(res => res.json())
            .then(data => setAnalytics(data));
    }, [restaurantId, dateRange]);
    
    if (!analytics) return <Loading />;
    
    return (
        <div className="analytics-grid">
            {/* KPIs */}
            <div className="kpi-cards">
                <KPICard 
                    title="Total Cash Handled"
                    value={`${analytics.total_cash_handled} TL`}
                    change="+12%"
                />
                <KPICard 
                    title="Alerts Generated"
                    value={analytics.alerts_count}
                    change="-8%"
                    changeType="positive"
                />
                <KPICard 
                    title="POS Compliance"
                    value={`${analytics.pos_compliance_rate}%`}
                />
                <KPICard 
                    title="Avg Transaction"
                    value={`${analytics.avg_transaction} TL`}
                />
            </div>
            
            {/* Hourly Cash Flow */}
            <div className="chart-container">
                <h3>Hourly Cash Flow</h3>
                <LineChart data={analytics.hourly_data}>
                    <XAxis dataKey="hour" />
                    <YAxis />
                    <Line type="monotone" dataKey="cash_handled" stroke="#8884d8" />
                    <Line type="monotone" dataKey="pos_recorded" stroke="#82ca9d" />
                </LineChart>
            </div>
            
            {/* Waiter Performance */}
            <div className="chart-container">
                <h3>Waiter Performance</h3>
                <BarChart data={analytics.waiter_stats}>
                    <XAxis dataKey="waiter_id" />
                    <YAxis />
                    <Bar dataKey="cash_handled" fill="#8884d8" />
                    <Bar dataKey="alerts" fill="#ff4444" />
                </BarChart>
            </div>
        </div>
    );
}
```

---

### EN: User Interface

React-based modern, responsive dashboard with real-time monitoring, event timeline, and analytics.

---

## ⚡ PERFORMANS VE ÖLÇEKLENDİRME / PERFORMANCE & SCALING

### TR: Sistem Performansı

#### Latency (Gecikme) Hedefleri:
```
Kamera → AI Detection: <100ms
AI Detection → Alert: <200ms
Alert → Dashboard: <500ms
Total End-to-End: <1 second
```

#### Throughput (İşlem Kapasitesi):
```
Tek GPU (T4):
- 25 FPS × 10 kamera = 250 frame/second
- Her frame ~10ms işleme süresi
- Toplam capacity: ~10-15 kamera per GPU

Production Setup (örnek):
- 100 restoran
- Her restoran 5 kamera
- Toplam 500 kamera
- Gerekli GPU: 500 / 10 = 50 GPU
- Yedeklilik ile: 60-70 GPU (AWS P3.2xlarge veya benzer)
```

#### Maliyet Hesaplama:
```
AWS Altyapısı (Aylık):

GPU İşleme (P3.2xlarge - Tesla V100):
- 50 instance × $3.06/hour × 730 hours = $111,690/month
- Spot instance ile %70 indirim = $33,507/month

Alternatif: Edge Computing
- Nvidia Jetson Xavier NX per restaurant
- $399 one-time per device
- 100 restaurant = $39,900 one-time
- Aylık cloud maliyeti: ~$5,000 (sadece video storage ve API)

Video Storage (S3):
- 500 kamera × 2 Mbps × 730 hours = ~330 TB/month
- S3 Standard: $7,590/month
- S3 Intelligent Tiering: ~$4,500/month (30 gün sonra arşiv)

Database (RDS PostgreSQL):
- db.r5.4xlarge: $1,800/month

Total Monthly Cost:
- Cloud-only: ~$40,000-$50,000/month
- Hybrid (Edge + Cloud): ~$10,000-$15,000/month (after initial hardware investment)
```

#### Ölçeklendirme Stratejisi:

**Faz 1: MVP (1-10 restoran)**
```
Infrastructure:
- 2x GPU instance (redundancy)
- Single region (örn: EU-West)
- PostgreSQL RDS
- Redis cache
- S3 for video storage

Cost: ~$2,000-$3,000/month
```

**Faz 2: Growth (10-100 restoran)**
```
Infrastructure:
- Kubernetes cluster (auto-scaling)
- 10-20 GPU instances
- Multi-region support
- CDN for video delivery
- Advanced analytics pipeline

Cost: ~$15,000-$25,000/month
```

**Faz 3: Scale (100-1000 restoran)**
```
Infrastructure:
- Hybrid Edge + Cloud
- Edge processing at restaurant level
- Cloud for aggregation and analytics
- Multi-region, multi-cloud
- Advanced ML ops pipeline

Cost: ~$50,000-$100,000/month (operational)
Initial Investment: ~$200,000-$500,000 (edge devices)
```

---

### EN: System Performance

Detailed latency targets, throughput calculations, cost breakdown, and scaling strategy for MVP through enterprise scale.

---

## 🔒 GÜVENLİK VE UYUMLULUK / SECURITY & COMPLIANCE

### TR: Güvenlik Mimarisi

#### 1. Video Şifreleme:
```
Transport Layer:
- TLS 1.3 for all API communication
- WSS (WebSocket Secure) for video streams
- RTSP over TLS for camera connections

Storage Layer:
- AES-256 encryption at rest (S3)
- Encrypted database (RDS with encryption)
- Key management: AWS KMS
```

#### 2. Erişim Kontrolü:
```python
# Role-based access control (RBAC)

ROLES = {
    'restaurant_owner': [
        'view:cameras',
        'view:events',
        'view:analytics',
        'manage:cameras',
        'manage:users'
    ],
    'manager': [
        'view:cameras',
        'view:events',
        'view:analytics',
        'acknowledge:alerts'
    ],
    'staff': [
        'view:own_events'  # Sadece kendi olaylarını görebilir
    ]
}

@app.get("/api/v1/events")
@require_permission('view:events')
async def get_events(current_user: User):
    if current_user.role == 'staff':
        # Sadece kendi olayları
        return filter_events_by_waiter(current_user.waiter_id)
    else:
        return get_all_events()
```

#### 3. KVKK/GDPR Uyumluluğu:

**Veri Saklama Politikası:**
```
Video Kayıtları:
- Hot storage: 7 gün (sık erişilen)
- Warm storage: 30 gün (orta erişim)
- Cold storage: 90 gün (arşiv)
- Otomatik silme: 90 gün sonra

Olay Verileri:
- Active data: 1 yıl
- Archive: 7 yıl (yasal zorunluluk)

Kişisel Veriler:
- Face blurring: Opsiyonel (privacy mode)
- Anonymization: Track ID kullanımı (isim yok)
- Right to deletion: Kullanıcı talebi üzerine veri silme
```

**Rıza Yönetimi:**
```python
# Çalışan rıza formu
CONSENT_FORM = {
    'video_monitoring': {
        'required': True,
        'description': 'Çalışma alanında video kayıt yapılacak',
        'retention': '90 gün'
    },
    'ai_analysis': {
        'required': True,
        'description': 'Yapay zeka ile hareket analizi yapılacak',
    },
    'data_sharing': {
        'required': False,
        'description': 'Anonim veriler analiz için kullanılabilir',
    }
}

# Her çalışan işe başlarken rıza formu imzalar
await consent_service.record_consent(
    employee_id=employee.id,
    consents=signed_consents,
    signed_at=datetime.now()
)
```

#### 4. Audit Logging:
```python
# Tüm kritik işlemler loglanır

@app.get("/api/v1/events/{event_id}/video")
async def get_event_video(event_id: str, current_user: User):
    # Audit log
    await audit_log.record({
        'action': 'VIDEO_ACCESS',
        'user_id': current_user.id,
        'resource': f'event:{event_id}',
        'timestamp': datetime.now(),
        'ip_address': request.client.host
    })
    
    return video_service.get_clip(event_id)
```

---

### EN: Security Architecture

Comprehensive security covering video encryption, access control, GDPR/KVKV compliance, consent management, and audit logging.

---

## 📊 MVP ÖZELLİKLERİ (1.5 AY) / MVP FEATURES (1.5 MONTHS)

### TR: Minimum Viable Product

#### Sprint 1 (Hafta 1-2): Temel Altyapı
- [x] Kamera entegrasyonu (RTSP)
- [x] Video frame yakalama
- [x] Frame preprocessing pipeline
- [x] Temel backend API (FastAPI)
- [x] PostgreSQL veritabanı setup

#### Sprint 2 (Hafta 3-4): AI Modelleri
- [x] YOLOv8 object detection entegrasyonu
- [x] Custom model training (para tespiti)
- [x] DeepSORT tracking entegrasyonu
- [x] Temel event detection logic

#### Sprint 3 (Hafta 5-6): Dashboard ve Demo
- [x] React frontend temel yapı
- [x] Live camera görünümü
- [x] Event timeline
- [x] Alert sistemi
- [x] Demo senaryosu hazırlama

#### MVP Demo Özellikleri:
```
✅ 1 kamera canlı izleme
✅ Para tespiti (Türk Lirası)
✅ Garson takibi
✅ Kasa alanı geo-fencing
✅ Gerçek zamanlı alert
✅ Basit dashboard (istatistikler)
✅ Video playback (olay anı)
✅ Responsive tasarım (tablet demo için)
```

---

### EN: Minimum Viable Product

3-sprint development plan with core features: camera integration, AI models, and demo-ready dashboard.

---

## 🎯 SONUÇ / CONCLUSION

### TR: Teknik Özet

ImageProcessingAI, **5 katmanlı mimari** ile restaurant payment monitoring sağlar:

1. **Edge Layer**: RTSP/IP kameralardan video akışı
2. **Processing**: OpenCV ile görüntü ön işleme
3. **AI Layer**: YOLOv8 (detection) + DeepSORT (tracking) + Event correlation
4. **Backend**: FastAPI mikroservisler + PostgreSQL + Redis + WebSocket
5. **Frontend**: React dashboard + real-time monitoring

**Temel Akış:**
```
Kamera → Frame → Preprocessing → YOLOv8 Detection → DeepSORT Tracking 
→ Event Correlation → Alert Generation → Dashboard Display → Video Archive
```

**Performans:**
- Latency: <1 second end-to-end
- Capacity: 10-15 cameras per GPU
- Accuracy: >95% detection rate (custom trained)

**1.5 Aylık MVP** teslim edebilir durumda ve yatırımcı demosu için hazır.

---

### EN: Technical Summary

ImageProcessingAI delivers restaurant payment monitoring through a 5