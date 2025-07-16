import cv2
import os

def extract_first_n_frames(video_path, output_dir, n_frames=30):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return False
    
    # 读取前n帧
    for i in range(n_frames):
        ret, frame = cap.read()
        if ret:
            # 保存帧为图片
            output_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"已保存第 {i+1} 帧到 {output_path}")
        else:
            print(f"警告：只能读取到 {i} 帧")
            break
    
    # 释放资源
    cap.release()
    return True

if __name__ == "__main__":
    video_path = os.path.join("alarms", "local_video_20250713_203133.mp4")
    output_dir = "extracted_frames"
    
    success = extract_first_n_frames(video_path, output_dir)
    if success:
        print("帧提取完成！")
    else:
        print("帧提取失败！") 