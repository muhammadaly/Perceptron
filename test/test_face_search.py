from face_functions.face_search_facescrub import FaceAuthenticator
import cv2

def test_video():

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(r"D:\dataset\data\test\YouCut_20200521_082238048.mp4")
    fa = FaceAuthenticator()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r"D:\dataset\data\test\out.avi", fourcc, 20.0, (644, 1144))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    ind = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # frame = cv2.resize(frame, (640, 480))
            # Display the resulting frame
            result = fa.search_mutiple(frame)
            cv2.imshow('Frame', result)
            out.write(result)

            ind += 1
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def test_img():
    fa = FaceAuthenticator()
    img0 = cv2.imread(r"D:\dataset\data\test\justin_timberlake_acusen_infiel.jpg")
    result = fa.search_mutiple(img0)
    cv2.imshow('result', result)
    cv2.waitKey()


if __name__ == '__main__':
    test_video()
