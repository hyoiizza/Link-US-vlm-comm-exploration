# 두 로버 실험 실행 순서 (linkus, 인터넷 없이 직접 실행)

linkus에는 인터넷이 없어서, robot1을 linkus에 연결하면 Claude도 멈춥니다. 이 문서대로 직접 진행하고,
실험이 끝나면 인터넷에 다시 연결한 뒤 Claude에게 분석을 요청하세요. 데이터는 `~/experiment_runs/`에 쌓입니다.

---

## 1. linkus로 전환

**robot2 (robot2의 화면·키보드에서 직접)**
```
nmcli connection up linkus
sudo iw dev wlP1p1s0 set power_save off
```

**robot1**
```
nmcli connection up linkus
iw dev wlP1p1s0 link | grep SSID          # SSID: linkus 확인
iw dev wlP1p1s0 get power_save            # Power save: off 확인 (아니면: sudo iw dev wlP1p1s0 set power_save off)
```

## 2. robot1에서 robot2 찾기 (IP를 자동으로 config.sh에 저장)
```
bash ~/experiment_kit/find_robot2.sh
```
마지막에 `ssh ok: robot2`, `SSID: linkus`가 나오면 성공입니다.

## 2-1. robot1에서 robot2로 코드·키트 보내기 (인터넷 불필요, 코드를 고친 뒤 한 번)
```
bash ~/experiment_kit/sync_robot2.sh
```
robot2에서 실제로 실행 중인 launch 확인·정리:
```
ssh gitae@192.168.0.202 'ps -eo args | grep "[r]os2 launch" || echo none'
ssh gitae@192.168.0.202 'bash ~/experiment_kit/ros_cleanup.sh'
```

## 3. 매 세션 시작 전 확인

- **공유기(AP) 위치:** linkus 공유기를 **아이폰이 있던 자리(robot1 왼쪽 뒷바퀴 옆 바닥)**에 두세요. 설정값이 이미 그 위치입니다 (`ap_x -0.25, ap_y 0.30`).
  다른 곳에 두면 `~/ros2_ws/src/rover_multi/config/team.yaml`의 `ap_x`, `ap_y`를 robot1 출발점 기준 좌표(m, 앞 +x, 왼쪽 +y)로 바꾸세요.
- **로봇 배치:** robot1은 출발 표시 위, robot2는 **robot1 오른쪽 0.6 m, 같은 방향** (`world.yaml`에 설정됨).
- **robot1 모터 장치:** `ls -l /dev/lx16a` 가 없으면
  ```
  sudo insmod ~/ch341_driver/ch341.ko
  ```
- **robot1 메모리:** Firefox, VS Code를 닫으세요 (VLM이 GPU 메모리 부족으로 안 켜질 수 있음).
- **목표(사람):** 모든 실행에서 같은 자리에 있고, 출발 위치에서는 보이지 않는 곳.

## 3-1. 매 실행 전 (자동으로 검사되지만 알아둘 것)
- **이전 실행을 반드시 종료한 뒤** 다음 실행을 시작하세요. 남아 있으면 라이다·IMU 포트를 두 번 열어서 둘 다 실패합니다.
  start_pair.sh가 이를 감지하면 시작을 거부합니다. 정리: `bash ~/experiment_kit/ros_cleanup.sh` (robot2는 `ssh gitae@$ROBOT2_HOST 'bash ~/experiment_kit/ros_cleanup.sh'`)
- **robot2를 linkus에서 재부팅했다면 시계가 1970년**이 됩니다 (배터리 없음, 인터넷 없음). start_pair.sh가 2초 이상 차이를 감지하면 거부합니다.
  맞추기 (robot2 sudo 비밀번호 입력): `bash ~/experiment_kit/set_robot2_time.sh`

## 4. 실험 1회 실행 (robot1 터미널)

```
# VLM (제안 방법): 목표가 robot1 출발점에서 앞으로 12 m, 오른쪽 2.5 m
bash ~/experiment_kit/start_pair.sh p_vlm_1 proposed 1 yolo "메모" --front 12 --right 2.5

# 기본 (Baseline)
bash ~/experiment_kit/start_pair.sh b_base_1 frontier 0 yolo "메모" --front 12 --right 2.5
```
- **목표 위치는 필수**입니다: `--front <m>`와 `--left <m>` 또는 `--right <m>`. robot1 중심 기준, robot1이 바라보는 방향이 앞입니다.
  줄자로 잰 값을 넣으세요. 분석할 때 이 위치에서 1.5 m 안에서 YOLO가 검출한 사람만 "발견"으로 칩니다 (다른 사람은 무시).
- 점검 결과가 나온 뒤 `Start run ...? [y/N]` 에 `y`를 입력하면 시작합니다.
- 실행 이름은 매번 다르게 (`p_vlm_1`, `b_base_1`, `p_vlm_2`, `b_base_2` ...). 같은 이름이면 거부됩니다.
- 순서는 **VLM과 Baseline을 번갈아** 하는 것을 권장합니다 (배터리·환경 변화가 한쪽에 몰리지 않게).
- 추가 비교(선택): `comm_aware 0`, `semantic_only 1`.

## 5. 주행 중 확인

robot1 화면에 **coordinator 로그 창**이 자동으로 뜹니다.

| 정상 | 문제 |
|---|---|
| `robot2 joined` | `robot2 lost` 가 계속 반복 |
| `Selection N: robot1 -> ...`, `robot2 -> ...` (두 로봇이 다른 골) | 1~2분이 지나도 `Selection` 이 없음 |
| `New person #...` (YOLO가 사람 확인) | `process has died` |

robot2 로그 보기 (robot1 터미널):
```
source ~/experiment_kit/config.sh
ssh gitae@$ROBOT2_HOST 'tail -f ~/experiment_runs/<실행이름>/robot2_launch.log | grep -E "team_agent|died|person"'
```

**디버깅 화면(RViz):** robot1에서 켜면 CPU를 약 25% 더 써서 robot1 지도 작성이 느려질 수 있습니다.
보고 싶으면 **robot2 화면에서** 띄우는 것을 권장합니다 (robot2 터미널):
```
source ~/ros2_ws/install/setup.bash && ros2 launch rover_multi team_rviz.launch.py
```
꼭 robot1에서 보려면 실행 앞에 `DEBUG_RVIZ=1`을 붙이세요: `DEBUG_RVIZ=1 bash ~/experiment_kit/start_pair.sh ...`

## 6. 종료 (사람을 찾았거나 15분이 지나면)
```
bash ~/experiment_kit/stop_pair.sh <실행이름>
```
녹화를 먼저 닫고 두 로봇을 멈춘 뒤 robot2의 파일을 robot1로 가져옵니다. `robot1 bag closed`, `robot2 bag closed`가 나오면 정상.

**이름을 모르거나 다 멈추고 싶을 때 (robot1에서, robot1이 linkus에 연결된 상태로):**
```
bash ~/experiment_kit/stop_all.sh
```
녹화를 먼저 닫고 두 로봇의 ROS를 모두 끈 뒤, 아직 안 가져온 robot2 기록을 robot1로 가져옵니다.
**robot1을 인터넷 Wi-Fi로 바꾸기 전에** 실행하세요 (바꾼 뒤에는 robot2에 닿지 않습니다).

**비상 정지**
- robot1: `pkill -INT -f "ros2 launch rover_multi"`
- robot2 (robot1에서): `source ~/experiment_kit/config.sh; ssh gitae@$ROBOT2_HOST 'bash ~/experiment_kit/robot2_stop.sh <실행이름>'`
- robot2 (robot2에서 직접): `bash ~/experiment_kit/robot2_stop.sh <실행이름>`
- 그래도 안 멈추면 모터 전원을 끄세요.

robot2와 연결이 끊긴 채로 종료했다면, 연결된 뒤 robot2 파일만 다시 가져오기:
```
source ~/experiment_kit/config.sh; mkdir -p ~/experiment_runs/<실행이름>/robot2
rsync -a gitae@$ROBOT2_HOST:experiment_runs/<실행이름>/ ~/experiment_runs/<실행이름>/robot2/
```

## 7. 실험이 끝나면
1. 두 로봇을 인터넷 Wi-Fi로 되돌립니다: `nmcli connection up AIRL_5G` (또는 아이폰)
2. Claude에게 "실험 끝났어, 분석해줘"라고 하면 표 1, 표 2, 그림 3을 만듭니다.
3. 실행마다 사람 위치를 기록해 두면 좋습니다 (YOLO 위치와 비교 검증용).

## 참고: 로그 위치
- `~/experiment_runs/<실행이름>/robot1_launch.log`, `robot2/robot2_launch.log`
- `manifest.yaml`: 방법, VLM, Wi-Fi 이름, 절전 모드, 두 로봇 시계 차이
