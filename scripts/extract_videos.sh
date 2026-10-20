#!/bin/bash
# Unpack the benchmark video archives. Video is already compressed, so the
# extracted footprint is roughly the same as the archives.
set -u
B=/data/cl195/NeurIPS_workshop/data/raw
mkdir -p $B/IntentBench/videos $B/Daily-Omni/videos $B/WorldSense/videos $B/WorldSense/subtitles
echo "[IntentBench]"; unzip -q -o $B/IntentBench/videos.zip -d $B/IntentBench/videos && echo "  ok"
echo "[Daily-Omni]";  tar -xf $B/Daily-Omni/Videos.tar -C $B/Daily-Omni/videos && echo "  ok"
echo "[WorldSense subtitles]"; unzip -q -o $B/WorldSense/worldsense_subtitles.zip -d $B/WorldSense/subtitles && echo "  ok"
for z in $B/WorldSense/worldsense_videos_*.zip; do
  echo "[WorldSense $(basename $z)]"; unzip -q -o "$z" -d $B/WorldSense/videos && echo "  ok"
done
echo "DONE"
