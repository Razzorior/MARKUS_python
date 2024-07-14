import zmq
import python_handler

model = None

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
ps = python_handler.PythonState()

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)
    decoded_message = message.decode('utf-8')
    response, is_json = ps.manage_request(decoded_message, socket)
    if is_json:
        # Check if there are multiple responses or not
        if isinstance(response, list):
            print('Detected JSON List to send')
            for i, message in enumerate(response):
                print('Sending {}. message'.format(i))
                more = i < len(response) - 1
                socket.send_json(message, flags=zmq.SNDMORE if more else 0)
        else:
            print('Sending Response..')
            socket.send_json(response)
    else:
        if isinstance(response, list):
            print('Detected List to send')
            for i, message in enumerate(response):
                more = i < len(response) - 1
                socket.send(message, flags=zmq.SNDMORE if more else 0)
        else:
            socket.send(response)