import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import windows
from scipy.signal import find_peaks
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve


# Function to preprocess data with zero-mean-and-scaling technique
def preprocess(data):
    scaled_data = (data - np.mean(data)) / np.std(data)
    return scaled_data


# Detrend using Smoothness Prior Approach (SPA)
def detrend(signal, window_size=10, overlap=0.5):
    detrended_signal = np.zeros_like(signal)
    step = int(window_size * (1 - overlap))

    hamming_window = windows.hamming(window_size)

    for i in range(0, len(signal) - window_size, step):
        window = signal[i:i + window_size]

        window_detrended = window - np.mean(window)
        window_detrended *= hamming_window
        detrended_signal[i:i + window_size] = window_detrended

    return detrended_signal


# Moving average filter
def moving_average(signal, window_size=5):
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, weights, mode='valid')
    smoothed_signal = np.pad(smoothed_signal, (window_size - 1, 0), mode='edge')
    return smoothed_signal


# Butterworth band-pass filter
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=8):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


# POS algorithm
def pos_algorithm(R, G, B, fs):
    # Calculate XS and YS
    XS = G - B
    YS = (-2 * R) + G + B

    # Calculate standard deviations
    L = int(1.6 * fs)  # Number of samples in 1.6 seconds of video
    sigma_XS = np.std(XS, ddof=1)
    sigma_YS = np.std(YS, ddof=1)

    # Calculate scaling factor alpha
    alpha = sigma_XS / sigma_YS

    # Obtain rPPG signal
    rPPG = XS + (alpha * YS)

    return rPPG


def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal


def two_step_wavelet_filter(signal, window_size=10, wavelet='db4', level=1, wavelet_window_size=4, wavelet_std=0.1, gaussian_std=0.1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    squared_coeffs = [np.square(np.abs(c)) for c in coeffs]

    gaussian_window = gaussian_filter1d(np.ones(wavelet_window_size), wavelet_std)
    smoothed_coeffs = [convolve(sc, gaussian_window, mode='same') for sc in squared_coeffs]

    max_scale_index = np.argmax([np.sum(sc) for sc in smoothed_coeffs])

    gaussian_filter = gaussian_filter1d(np.ones(window_size), gaussian_std)
    smoothed_coeffs[max_scale_index] = convolve(smoothed_coeffs[max_scale_index], gaussian_filter, mode='same')

    filtered_signal = pywt.waverec(smoothed_coeffs, wavelet)

    return filtered_signal


def estimate_heart_rate(peaks, fs):
    peak_times = peaks / fs
    rr_intervals = np.diff(peak_times)

    heart_rate_bpm = 60 / np.mean(rr_intervals)
    return heart_rate_bpm


xds = 0
heart_rate_bpm = 0
cap = cv2.VideoCapture(0)

red_means_preprocessed = []
green_means_preprocessed = []
blue_means_preprocessed = []

fs = 28

while True:

    ret, frame = cap.read()
    if not ret:
        break
    xds = xds + 1

    if xds > 100:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:

            forehead_roi = frame[y+int(y*0.05):y + int(0.25 * h), x + w - int(0.75 * w):x + int(0.75 * w)]

            spatial_avg = np.mean(forehead_roi, axis=(0, 1))

            red_mean = spatial_avg[0]
            green_mean = spatial_avg[1]
            blue_mean = spatial_avg[2]

            red_mean = (red_mean/len(forehead_roi))*100
            green_mean = (green_mean/len(forehead_roi))*100
            blue_mean = (blue_mean/len(forehead_roi))*100

            red_means_preprocessed = np.append(red_means_preprocessed, red_mean)
            green_means_preprocessed = np.append(green_means_preprocessed, green_mean)
            blue_means_preprocessed = np.append(blue_means_preprocessed, blue_mean)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x + w - int(0.75 * w), y+int(y*0.05)), (x + int(0.75 * w), y + int(0.25 * h)), (0, 255, ), 1)
            cv2.circle(frame, (x + int(w * 0.25), int(y + int(h * 0.6))), int(w * 0.10), (0, 255, 0), 1)
            cv2.circle(frame, (x + int(w * 0.77), int(y + int(h * 0.6))), int(w * 0.10), (0, 255, 0), 1)
            cv2.putText(frame, f'Heart Rate (BPM) : {heart_rate_bpm}', (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Face', frame)

        if xds % (fs * 10) == 0:

            red_means_preprocessed = np.array(red_means_preprocessed)
            green_means_preprocessed = np.array(green_means_preprocessed)
            blue_means_preprocessed = np.array(blue_means_preprocessed)

            plt.plot(red_means_preprocessed, label='Red', color='red')
            plt.plot(green_means_preprocessed, label='Green',color='green')
            plt.plot(blue_means_preprocessed, label='Blue',color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title('Data')
            plt.legend()
            plt.show()

            red_means_scaled = preprocess(red_means_preprocessed)
            green_means_scaled = preprocess(green_means_preprocessed)
            blue_means_scaled = preprocess(blue_means_preprocessed)

            plt.plot(red_means_scaled, label='Red', color='red')
            plt.plot(green_means_scaled, label='Green', color='green')
            plt.plot(blue_means_scaled, label='Blue', color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title('Scaled')
            plt.legend()
            plt.show()

            red_means_detrended = detrend(red_means_scaled)
            green_means_detrended = detrend(green_means_scaled)
            blue_means_detrended = detrend(blue_means_scaled)

            plt.plot(red_means_detrended, label='Red', color='red')
            plt.plot(green_means_detrended, label='Green', color='green')
            plt.plot(blue_means_detrended, label='Blue', color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title('Detrended')
            plt.legend()
            plt.show()

            window_size = 5
            red_means_smoothed = moving_average(red_means_detrended, window_size=window_size)
            green_means_smoothed = moving_average(green_means_detrended, window_size=window_size)
            blue_means_smoothed = moving_average(blue_means_detrended, window_size=window_size)

            plt.plot(red_means_smoothed, label='Red', color='red')
            plt.plot(green_means_smoothed, label='Green', color='green')
            plt.plot(blue_means_smoothed, label='Blue', color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title('Moving Average')
            plt.legend()
            plt.show()

            lowcut = 0.5
            highcut = 4.0
            red_means_filtered = butter_bandpass_filter(red_means_smoothed, lowcut, highcut, fs)
            green_means_filtered = butter_bandpass_filter(green_means_smoothed, lowcut, highcut, fs)
            blue_means_filtered = butter_bandpass_filter(blue_means_smoothed, lowcut, highcut, fs)

            plt.plot(red_means_filtered, label='Red')
            plt.plot(green_means_filtered, label='Green')
            plt.plot(blue_means_filtered, label='Blue')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title('Butterworth Filter')
            plt.legend()
            plt.show()

            pos_rPPG_signal = pos_algorithm(red_means_filtered, green_means_filtered, blue_means_filtered, fs)

            time_sec = np.arange(len(pos_rPPG_signal)) / fs
            plt.plot(time_sec, pos_rPPG_signal, label='Signal', color='blue')
            plt.xlabel('Time(s)')
            plt.ylabel('Value')
            plt.title('rPPG signal')
            plt.legend()
            plt.show()

            pos_rPPG_signal_normalized = normalize_signal(pos_rPPG_signal)

            time_sec = np.arange(len(pos_rPPG_signal_normalized)) / fs
            plt.plot(time_sec, pos_rPPG_signal_normalized, label='Signal', color='blue')
            plt.xlabel('Time(s)')
            plt.ylabel('Value')
            plt.title('rPPG normalized signal')
            plt.legend()
            plt.show()

            lowcut = 0.5
            highcut = 4.0

            filtered_signal = two_step_wavelet_filter(pos_rPPG_signal_normalized)

            time_sec = np.arange(len(filtered_signal)) / fs
            plt.plot(time_sec, filtered_signal, label='Filtered Signal', color='green')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Filtered rPPG Signal')
            plt.legend()
            plt.grid(True)
            plt.show()

            peaks, _ = find_peaks(pos_rPPG_signal_normalized, distance=int(fs * 0.5), height=0.5)

            if len(peaks) >= 2:
                heart_rate_bpm = estimate_heart_rate(peaks, fs)
                print("Estimated Heart Rate (BPM):", heart_rate_bpm)

            red_means_preprocessed = []
            green_means_preprocessed = []
            blue_means_preprocessed = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
