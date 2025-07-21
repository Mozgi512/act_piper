import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# XMLモデルファイルをロード
try:
    model = mujoco.MjModel.from_xml_path('bimanual_piper_ee_transfer_cube.xml')
except FileNotFoundError:
    exit()
data = mujoco.MjData(model)

# mocapオブジェクトのIDを取得
# XMLで定義した名前からIDを探します
mocap_left_id = model.body('l_mocap').mocapid[0]
mocap_right_id = model.body('r_mocap').mocapid[0]
left_hand_id = model.body('l_weldpoint').id
right_hand_id = model.body('r_weldpoint').id

# === グラフ描画用のデータ格納リスト ===
times = []
left_errors = []
right_errors = []

# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
  start_time = time.time()

  while viewer.is_running():
    step_start = time.time()
    current_time = data.time
    elapsed_time = time.time() - start_time
    
    if(elapsed_time < 10):
        # 左手mocapの目標位置 (Y-Z平面で円運動)
        left_target_pos = np.array([
            -0.1, # X座標 (前方)
            0.1 * np.cos(1 * current_time), # Y座標
            0.2 + 0.1 * np.sin(1 * current_time)  # Z座標
        ])
        
        # 右手mocapの目標位置 (Y-Z平面で逆位相の円運動)
        right_target_pos = np.array([
            0.1, # X座標 (前方)
            - 0.1 * np.cos(5 * current_time), # Y座標
            0.2 + 0.1 * np.sin(5 * current_time)   # Z座標
        ])
        
        # 計算した目標位置をmocapに設定
        data.mocap_pos[mocap_left_id] = left_target_pos
        data.mocap_pos[mocap_right_id] = right_target_pos
    
    # シミュレーションを1ステップ進める
    mujoco.mj_step(model, data)

    # === 誤差計算 ===
    left_hand_pos = data.body(left_hand_id).xpos
    right_hand_pos = data.body(right_hand_id).xpos
    
    left_mocap_pos = data.mocap_pos[mocap_left_id]
    right_mocap_pos = data.mocap_pos[mocap_right_id]

    error_left = np.linalg.norm(left_hand_pos - left_mocap_pos)
    error_right = np.linalg.norm(right_hand_pos - right_mocap_pos)
    
    times.append(current_time)
    left_errors.append(error_left)
    right_errors.append(error_right)

    # ビューアを更新して描画
    viewer.sync()
    
    # シミュレーションを指定したタイムステップに合わせるための待機
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
# === シミュレーション終了後、グラフを作成・保存 ===
print("シミュレーション終了。誤差グラフを作成します...")

plt.figure(figsize=(12, 7))
plt.plot(times, left_errors, label='Left Arm Error', color='royalblue')
plt.plot(times, right_errors, label='Right Arm Error', color='crimson', linestyle='--')

plt.title('Mocap-Hand Tracking Error vs. Time', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Euclidean Distance Error (m)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_filename = 'mocap_hand_error3.png'
plt.savefig(plot_filename)

print(f"グラフを '{plot_filename}' として保存しました。")  