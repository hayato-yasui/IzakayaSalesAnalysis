import os.path
import openpyxl as px

class Util:

    @staticmethod
    def create_dir(path):
        os.path.exists(os.mkdir(path))

    @staticmethod
    def check_existing_and_create_excel_file(file_path):
        if not os.path.exists(file_path):
            wb = px.Workbook()
            wb.save(file_path)

    @staticmethod
    def df_to_csv(df, dir, file_name,index=False):
        if not os.path.exists(dir):
            os.mkdir(dir)
        df.to_csv(dir + '/' + file_name, encoding='cp932',index=index)

