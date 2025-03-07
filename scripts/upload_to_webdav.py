"""
WebDAV上传工具

使用方法:
  python scripts/upload_to_webdav.py <path> [-c <config_file>] [-p]

参数:
  path: 要上传的本地文件或目录路径
  -c, --config: 配置文件路径 (默认为 'scripts/webdav_config.ini')
  -p, --package: 是否将目录打包成zip文件上传

配置文件 (webdav_config.ini) 示例:
  [webdav]
  webdav_hostname = your_webdav_hostname
  webdav_login = your_webdav_login (可选)
  webdav_password = your_webdav_password (可选)
  disable_check = True/False (可选, 默认为 True, 禁用SSL证书验证)
  root_path = /your/root/path (可选, 默认为 '/')
  ssl_verify = True/False (可选, 默认为 False, 启用/禁用 SSL 证书验证)

示例:
  1. 上传文件: python scripts/upload_to_webdav.py /path/to/your/file.txt
  2. 上传目录: python scripts/upload_to_webdav.py /path/to/your/directory
  3. 上传目录并打包成zip: python scripts/upload_to_webdav.py /path/to/your/directory -p
  4. 使用自定义配置文件: python scripts/upload_to_webdav.py /path/to/your/file.txt -c /path/to/your/custom_config.ini
"""
import os
import sys
import argparse
import configparser
import zipfile
import datetime
from webdav3.client import Client
from webdav3.exceptions import WebDavException
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_config(config_path='webdav_config.ini'):
    """加载并验证WebDAV配置"""
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"配置文件 '{config_path}' 未找到")
    
    required_keys = ['webdav_hostname']
    for key in required_keys:
        if not config.has_option('webdav', key):
            raise ValueError(f"配置文件中缺少必须字段: {key}")

    options = {
        'webdav_hostname': config.get('webdav', 'webdav_hostname'),
        'webdav_login': config.get('webdav', 'webdav_login', fallback=''),
        'webdav_password': config.get('webdav', 'webdav_password', fallback=''),
        'disable_check': config.getboolean('webdav', 'disable_check', fallback=True),  # 新增关键参数
        'root_path': config.get('webdav', 'root_path', fallback='/').strip('/')
    }
    
    if options['root_path']:
        options['root_path'] = f"/{options['root_path']}/"
    else:
        options['root_path'] = '/'
    
    ssl_verify = config.getboolean('webdav', 'ssl_verify', fallback=False)
    return options, ssl_verify

def verify_connection(client, root_path):
    """替代check方法的连接验证"""
    try:
        # 尝试列出根目录内容
        client.list(root_path)
        return True
    except WebDavException as e:
        if "404" in str(e):
            raise RuntimeError(f"根路径不存在: {root_path}")
        elif "401" in str(e):
            raise RuntimeError("认证失败，请检查用户名密码")
        else:
            raise RuntimeError(f"连接验证失败: {str(e)}")

def upload_directory(client, options, local_dir):
    """增强版目录上传"""
    remote_root = os.path.join(options['root_path'], os.path.basename(local_dir)).replace('//', '/')
    
    # 创建目标目录
    try:
        client.mkdir(remote_root)
        print(f"✅ 创建目标目录: {remote_root}")
    except WebDavException as e:
        if "409" not in str(e):  # 忽略已存在的错误
            raise RuntimeError(f"目录创建失败: {str(e)}")

    for root, dirs, files in os.walk(local_dir):
        relative_path = os.path.relpath(root, local_dir)
        current_remote_dir = os.path.join(remote_root, relative_path).replace('\\', '/')

        # 创建子目录
        if not client.check(current_remote_dir):
            try:
                client.mkdir(current_remote_dir)
            except WebDavException as e:
                if "409" not in str(e):
                    raise

        # 上传文件
        for file in files:
            local_path = os.path.join(root, file)
            remote_path = os.path.join(current_remote_dir, file).replace('\\', '/')
            try:
                client.upload_sync(remote_path=remote_path, local_path=local_path)
                print(f"⬆️ 上传成功: {remote_path}")
            except Exception as e:
                raise RuntimeError(f"文件上传失败: {remote_path} => {str(e)}")

def create_zip(dir_path, zip_name):
    """将目录打包成zip文件"""
    zip_path = os.path.join(os.getcwd(), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                zipf.write(file_path, arcname)
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="WebDAV上传工具")
    parser.add_argument('path', help="要上传的本地文件或目录路径")
    parser.add_argument('-c', '--config', default='scripts/webdav_config.ini', help="配置文件路径")
    parser.add_argument('-p', '--package', action='store_true', help="是否将目录打包成zip文件上传")
    args = parser.parse_args()

    try:
        # 加载配置
        options, ssl_verify = load_config(args.config)
        
        # 初始化客户端
        client = Client({
            'webdav_hostname': options['webdav_hostname'],
            'webdav_login': options['webdav_login'],
            'webdav_password': options['webdav_password'],
            'disable_check': options['disable_check']
        })
        client.verify = ssl_verify
        client.default_options['disable_check'] = options['disable_check']

        # 连接验证
        verify_connection(client, options['root_path'])

        # 处理上传
        local_path = os.path.abspath(args.path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"路径不存在: {local_path}")

        if os.path.isfile(local_path):
            remote_path = os.path.join(options['root_path'], os.path.basename(local_path)).replace('//', '/')
            client.upload_sync(remote_path=remote_path, local_path=local_path)
            print(f"✅ 文件上传成功: {remote_path}")
        else:
            if args.package:
                # 打包目录
                dir_name = os.path.basename(local_path)
                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                zip_name = f"{dir_name}_{current_time}.zip"
                zip_path = create_zip(local_path, zip_name)

                # 上传zip文件
                remote_path = os.path.join(options['root_path'], zip_name).replace('//', '/')
                client.upload_sync(remote_path=remote_path, local_path=zip_path)
                print(f"✅ ZIP文件上传成功: {remote_path}")

                # 删除本地zip文件
                os.remove(zip_path)
                print(f"🗑️ 已删除本地ZIP文件: {zip_path}")

            else:
                upload_directory(client, options, local_path)
            
        print("🎉 上传完成！")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
