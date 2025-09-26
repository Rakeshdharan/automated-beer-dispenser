from picamera2 import Picamera2

picam2 = Picamera2()

main = {'size':(1640, 1232)}
raw = {'size':(1640, 1232)}
controls = {'FrameRate': 20}
config = picam2.create_video_configuration(main, raw=raw, controls=controls)
picam2.configure(config)

picam2.start_and_record_video('test_video_1.mp4', duration=20)
