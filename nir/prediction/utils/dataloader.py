import pymysql
import copy
import algorism.strdeal as strd
from itertools import zip_longest

class mysqlHelper:
    def __init__(self):
        self.cursor = 0

    def connect(self):
        # 连接 MySQL 数据库
        self.conn = pymysql.connect(
            host='Localhost',  # 主机名
            port=3306,         # 端口号，MySQL默认为3306
            user='root',       # 用户名
            password='123456',   # 密码
            database='nirdata',   # 数据库名称
        )

        # 使用 cursor() 方法创建一个游标对象 cursor，默认返回元素数据类型
        self.cursor = self.conn.cursor()

    # 获取光谱仪设备信息
    def getdevicedata(self):
        try:
            # 使用 execute()  方法执行 SQL 查询
            self.cursor.execute("select * from device_info")
            # 使用 fetchone() 方法获取单条数据.
            data = self.cursor.fetchone()
            # 输出data数据
            size = data[2] # 采样点个数
            wave = data[3] # 波长段
            dark = data[4] # 暗底
            return size, wave, dark
        except:
            print('Try Error -- Find device info failed.')

    # 获取标准光谱信息
    def getstandarddata(self):
        try:
            dict = {}
            # 使用 execute()  方法执行 SQL 查询
            self.cursor.execute("select * from standard_info")
            # 使用 fetchall() 方法获取所有数据.
            data = self.cursor.fetchall()
            # 输出data数据
            for row in data:
                stand = row[2] # 标准光谱
                intime = row[4] # 积分时间
                dict[intime] = stand
            return copy.deepcopy(dict) # 深拷贝，使返回值与函数体内变量完全独立
        except:
            print('Try Error -- Find standard info failed.')

    def getsamplehumid(self, id):
        try:
            sqlstr = "select * from sample_info where id=" + str(id)
            # 使用 execute()  方法执行 SQL 查询
            self.cursor.execute(sqlstr)
            # 使用 fetchone() 方法获取单条数据.
            data = self.cursor.fetchone()
            # 输出data数据
            humid = data[2] # 湿度值数据
            return humid
        except:
            print('Try Error -- Find humidity info failed.')

    def getsampledataall(self):
        try:
            sampleidgp = []
            humidgp = []
            # 使用 execute()  方法执行 SQL 查询
            self.cursor.execute("select * from sample_info")
            # 使用 fetchall() 方法获取所有数据.
            data = self.cursor.fetchall()
            # 输出data数据
            for row in data:
                sampleid = row[0] # 样本id号
                humid = row[2] # 湿度值
                sampleidgp.append(sampleid)
                humidgp.append(humid)
            return copy.deepcopy(sampleidgp), copy.deepcopy(humidgp) # 深拷贝，使返回值与函数体内变量完全独立
        except:
            print('Try Error -- Find sample info failed.')

    def getintensitydata(self, sampleid):
        try:
            intensgp = []
            intimegp = []
            # 使用 execute()  方法执行 SQL 查询
            sqlstr =  "select * from spectrum_info where sample=" + str(sampleid)
            self.cursor.execute(sqlstr)
            # 使用 fetchall() 方法获取所有数据.
            data = self.cursor.fetchall()
            # 输出data数据
            for row in data:
                intens = row[3] # 光谱数据
                intime = row[5] # 积分时间
                intensgp.append(strd.splitstr(intens, ',', 'int'))
                intimegp.append(intime)
            return copy.deepcopy(intensgp), copy.deepcopy(intimegp) # 深拷贝，使返回值与函数体内变量完全独立
        except:
            print('Try Error -- Find spectrum info failed.')

    def getintensdataall(self):
        try:
            humidgp = []
            intensgp = []
            intimegp = []
            # 使用 execute()  方法执行 SQL 查询
            self.cursor.execute("select * from spectrum_info")
            # 使用 fetchall() 方法获取所有数据.
            data = self.cursor.fetchall()
            # 输出data数据
            for row in data:
                sampleid = row[1] # 样本id号
                humid = self.getsamplehumid(sampleid) # 通过样本id查找湿度值
                intens = row[3] # 光谱数据
                intime = row[5] # 积分时间
                intensgp.append(strd.splitstr(intens, ',', 'int'))
                intimegp.append(intime)
                humidgp.append(humid)
            return copy.deepcopy(intensgp), copy.deepcopy(intimegp), copy.deepcopy(humidgp) # 深拷贝，使返回值与函数体内变量完全独立
        except:
            print('Try Error -- Find all spectrum info failed.')

    def close(self):
        self.cursor.close() # 关闭游标
        self.conn.close()   # 关闭数据库

class Dataloader:
    def __init__(self):
        self.sqlHelper = mysqlHelper()
        self.standict = {}

        # 连接数据库
        self.sqlHelper.connect()
        self.ldbaseinfo()
        self.ldstandinfo()

    # 获取基础信息（字段、波长、暗底）
    def ldbaseinfo(self):
        # 光谱仪设备信息
        self.size, _wave, _dark = self.sqlHelper.getdevicedata()
        # 字符串转 list
        self.wave = strd.splitstr(_wave, ',', 'float')
        self.dark = strd.splitstr(_dark, ',', 'int')

    # 获取标准光谱信息
    def ldstandinfo(self):
        # 标准光谱信息
        _standict = self.sqlHelper.getstandarddata()
        # 字符串转 list
        for key, value in _standict.items():
            _stand = strd.splitstr(value, ',', 'int')
            self.standict[key] = _stand

    # 获取样本信息
    def ldsampleinfo(self):
        samidgp, humidgp = self.sqlHelper.getsampledataall()
        data = zip(samidgp, humidgp)
        return data
    
    # 通过样本id加载光谱数据
    def ldintensinfo(self, sampleid, dark = False, stand = False):
        intensgp, intimegp = self.sqlHelper.getintensitydata(sampleid)
        humid = [self.sqlHelper.getsamplehumid(sampleid)]
        if (dark == True): # 去除暗底
            for i in range(len(intensgp)):
                for j in range(len(intensgp[i])): # 实际光强 = 当前光强 - 暗底
                    intensgp[i][j] = intensgp[i][j] - self.dark[j]
        if (stand == True): # 求反射率
            for i in range(len(intensgp)):
                for j in range(len(intensgp[i])): # 反射率 = 实际光强 / 标准光强（积分时间需对应）
                    intensgp[i][j] = intensgp[i][j] / (self.standict[intimegp[i]][j] - self.dark[j])
        '''for j in range(len(intensgp[0])): # 求平均值
            intens = 0
            for i in range(len(intensgp)):
                intens += intensgp[i][j]
            intens = intens / len(intensgp)
            for i in range(len(intensgp)):
                intensgp[i][j] = intens'''
        datagp = zip_longest(intensgp, intimegp, humid)
        return datagp

    # 获取所有光谱数据
    def ldintensallinfo(self, dark = False, stand = False, save = False):
        intensgp, intimegp, humidgp = self.sqlHelper.getintensdataall()
        if (dark == True): # 去除暗底
            for i in range(len(intensgp)):
                for j in range(len(intensgp[i])): # 实际光强 = 当前光强 - 暗底
                    intensgp[i][j] = intensgp[i][j] - self.dark[j]
        if (stand == True): # 求反射率
            for i in range(len(intensgp)):
                for j in range(len(intensgp[i])): # 反射率 = 实际光强 / 标准光强（积分时间需对应）
                    intensgp[i][j] = intensgp[i][j] / (self.standict[intimegp[i]][j] - self.dark[j])
        if (save == True): # 保存数据为 .txt
            with open('total.txt', 'w') as f:
                for i in range(len(intensgp)):
                    strline = ''
                    for j in range(len(intensgp[i])):
                        strline += '{:.4f}'.format(intensgp[i][j]) + ' '
                    strline += str(humidgp[i])
                    f.write(strline + '\n')
        datagp = zip(intensgp, intimegp, humidgp)
        return datagp
    
    # 获取 波长1430nm 的采样id号（172 -- 256）
    def get_wave1430_id(self):
        index = 0
        minln = 100000
        for i in range(len(self.wave)):
            if (abs(self.wave[i] - 1430) < minln):
                minln = abs(self.wave[i] - 1430)
                index = i
        return index

    # 获取训练集
    def loadtrain(self):
        data = []
        with open('./datasets/train.txt', mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                linedata = line.split() # 字符串中分割数字
                data.append(linedata[0])
            f.close()
        return data

    # 获取验证集
    def loadval(self):
        data = []
        with open('./datasets/val.txt', mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                linedata = line.split() # 字符串中分割数字
                data.append(linedata[0])
            f.close()
        return data

    # 获取测试集
    def loadtest(self):
        data = []
        with open('./datasets/test.txt', mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                linedata = line.split() # 字符串中分割数字
                data.append(linedata[0])
            f.close()
        return data

    def get_baseinfo(self):
        return self.size, self.wave, self.dark

    def get_standinfo(self):
        return self.standict
