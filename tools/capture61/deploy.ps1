param([string]$JetsonIp = '172.19.18.130')
$ErrorActionPreference = 'Stop'
$archive = Join-Path $PSScriptRoot 'capture61-overlay.tar.gz'
$remote = 'jetson@' + $JetsonIp
& ssh -o BatchMode=yes -o ConnectTimeout=8 $remote 'hostname'
if ($LASTEXITCODE -ne 0) { throw 'Jetson SSH unavailable; nothing launched.' }
& scp -o ConnectTimeout=8 $archive "${remote}:/home/jetson/deploy_staging/capture61-overlay.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
$installScript = @'
set -e
cd /home/jetson/deploy_staging
mkdir -p b18000-capture61-20260921
if [ ! -d b18000-capture61-20260921/mydog_policy ]; then
  cp -a b18000-no-extra-compensation-20260920/mydog_policy b18000-capture61-20260921/
fi
tar -xzf capture61-overlay.tar.gz -C b18000-capture61-20260921
source /opt/ros/humble/setup.bash
cd b18000-capture61-20260921/mydog_policy
export PYTHONPATH=.:$PYTHONPATH
python3 -m pytest test/test_capture61.py test/test_capture61_node.py test/test_capture61_plans.py test/test_rs01_model18000_node_offline.py test/test_b18000_supported_start.py test/test_b18000_sequence.py -q
cd ..
colcon build --base-paths mydog_policy --packages-select mydog_policy --symlink-install
echo 'Build complete. No controller, motors or motion started.'
'@
& ssh -o BatchMode=yes -o ConnectTimeout=8 $remote $installScript
if ($LASTEXITCODE -ne 0) { throw 'Offline verification or build failed; do not start candidate.' }
