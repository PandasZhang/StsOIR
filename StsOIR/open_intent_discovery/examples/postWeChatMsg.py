# import sys, json

# print(sys.argv)
# # print(json.dumps(sys.argv))
import argparse, requests
def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='M201-09-202201', help="Msg Name")
    parser.add_argument('--title', type=str, default='Pargram Finish', help="Msg Content")
    parser.add_argument('--content', type=str, default='Pargram Finish', help="Msg Content")
    parser.add_argument('--url', type=str, default='https://www.autodl.com/api/v1/wechat/message/push', help="Msg Url")
    parser.add_argument('--token', type=str, default='', help="Msg Content")
    args = parser.parse_args()
    return args

def postMsg(args):
    resp = requests.post(args.url, json=checkMsg(args))
    return resp

def checkMsg(args):
    jsonItem = {}
    jsonItem['token'] = args.token
    jsonItem['name'] = args.name
    jsonItem['title'] = args.title
    jsonItem['content'] = args.content
    return jsonItem

if __name__ == '__main__':
    args = parse_arguments()
    resp = postMsg(args)
    # print(args)
    print('======='*20, '\n')
    print(resp.content.decode(), '\n')
    print('======='*20)
