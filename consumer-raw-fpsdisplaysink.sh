gst-launch-1.0 shmsrc socket-path=/tmp/feed.raw do-timestamp=1 ! \
	video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! \
	identity sync=true ! \
	queue max-size-buffers=2 leaky=downstream ! \
	videoconvert ! \
	fpsdisplaysink video-sink=autovideosink text-overlay=true sync=false