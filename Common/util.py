import os.path


class Util:

    @staticmethod
    def create_dir(path):
        os.path.exists(os.mkdir(path))

    @staticmethod
    def df_to_csv(df, dir, file_name,index=False):
        if not os.path.exists(dir):
            os.mkdir(dir)
        df.to_csv(dir + '/' + file_name, encoding='cp932',index=index)

