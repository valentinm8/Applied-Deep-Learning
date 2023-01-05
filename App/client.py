import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = "82c21ca1b1ce"
PORT = 1236
s.connect((HOST, PORT))
msg = s.recv(1024)
print(msg.decode("utf-8"))