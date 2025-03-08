import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def get_heart_rate(signal, fps, min_heart_rate=40, max_heart_rate=180):
    """Calculate heart rate from the processed signal."""
    # Convert min and max HR to Hz
    min_hz = min_heart_rate / 60
    max_hz = max_heart_rate / 60
    
    # Compute FFT
    L = len(signal)
    signal = signal - np.mean(signal)  # Remove DC component
    
    # Apply window function to reduce spectral leakage
    windowed_signal = signal * np.hamming(L)
    
    # Compute FFT and frequency axis
    Y = fft(windowed_signal)
    Y = Y[:L//2]  # Take only the first half (positive frequencies)
    freqs = np.linspace(0, fps/2, L//2)
    
    # Find peaks in the frequency range corresponding to heart rates
    valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))
    valid_freqs = freqs[valid_idx]
    valid_fft = np.abs(Y[valid_idx])
    
    # Find the peak
    if len(valid_fft) > 0:
        max_idx = np.argmax(valid_fft)
        hr_freq = valid_freqs[max_idx]
        hr = hr_freq * 60  # Convert Hz to BPM
        return hr, valid_freqs, valid_fft
    else:
        return None, valid_freqs, valid_fft


def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set video parameters
    fps = 30  # frames per second
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Parameters for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Buffer to store green channel values
    buffer_size = 300  # 10 seconds at 30 fps
    green_values = []
    
    # Create a plot to display results
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    signal_line, = ax1.plot([], [], 'g-')
    fft_line, = ax2.plot([], [], 'r-')
    
    ax1.set_title('Green Channel Signal')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Value')
    
    ax2.set_title('FFT Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    
    hr_text = ax2.text(0.5, 0.9, '', transform=ax2.transAxes, ha='center')
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
            
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Define forehead region (upper middle part of the face)
                forehead_y = int(y + h * 0.2)
                forehead_h = int(h * 0.15)
                forehead_x = int(x + w * 0.3)
                forehead_w = int(w * 0.4)
                
                # Draw rectangle around forehead
                cv2.rectangle(frame, (forehead_x, forehead_y), 
                              (forehead_x+forehead_w, forehead_y+forehead_h), 
                              (0, 255, 0), 2)
                
                # Extract ROI
                roi = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]
                
                if roi.size > 0:
                    # Calculate the mean green value in the ROI
                    mean_green = np.mean(roi[:, :, 1])
                    green_values.append(mean_green)
                    
                    # Keep buffer at specified size
                    if len(green_values) > buffer_size:
                        green_values.pop(0)
                    
                    # Process the signal once we have enough data
                    if len(green_values) >= buffer_size * 0.9:
                        # Apply bandpass filter (0.7-3.5 Hz corresponds to 42-210 BPM)
                        filtered_values = butter_bandpass_filter(
                            np.array(green_values), 0.7, 3.5, fps, order=3)
                        
                        # Calculate heart rate
                        hr, freqs, fft_values = get_heart_rate(filtered_values, fps)
                        
                        # Update plots
                        signal_line.set_data(range(len(filtered_values)), filtered_values)
                        ax1.relim()
                        ax1.autoscale_view()
                        
                        if hr is not None:
                            fft_line.set_data(freqs, fft_values)
                            ax2.relim()
                            ax2.autoscale_view()
                            hr_text.set_text(f'Heart Rate: {hr:.1f} BPM')
                            
                            # Display heart rate on video frame
                            cv2.putText(frame, f"Heart Rate: {hr:.1f} BPM", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, (0, 255, 0), 2)
                
            # Display the frame
            cv2.imshow('Heart Rate Detection', frame)
            
            # Update the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Calculate actual FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                actual_fps = frame_count / elapsed_time
                print(f"Actual FPS: {actual_fps:.2f}")
                start_time = time.time()
                frame_count = 0
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        plt.close()


if __name__ == "__main__":
    main()