import pandas as pd
from utils import clean
from dataclasses import dataclass


@dataclass
class Caso(object):
    """Clase que contiene atributos de una descripcion para
    la caja

    """
    descripcion: str
    responsabilidad: str
    idx_caso: str

    def set_responsabilidad(self, responsabilidad):
        self.responsabilidad = responsabilidad

    def get_responsabilidad(self):
        return self.responsabilidad

    def set_descripcion(self, descripcion):
        self.descripcion = descripcion

    def get_descripcion(self):
        return self.descripcion

    def set_idx_caso(self, idx_caso):
        self.idx_caso = idx_caso

    def get_idx_caso(self):
        return self.idx_caso

    @staticmethod
    def excel2caso(name_excel, col_name_descripcion,
                   col_name_responsabilidad, _id=None):
        """Convierte el par de columnas en un objeto caso

        :param name_excel: str, path al excel
        :param col_name_descripcion: str, nombre de la columna
        :param col_name_responsabilidad: str, nombre de la columna
        :param _id: str, para el seguimiento
        :returns: list, lista de casos instanciados.

        """
        df = pd.read_excel(name_excel)
        df.columns = map(str.lower, df.columns)

        df.dropna(
            subset=[col_name_descripcion, col_name_responsabilidad],
            inplace=True)
        df.drop_duplicates(
            subset=[col_name_descripcion, col_name_responsabilidad],
            inplace=True)
        df.reset_index(inplace=True, drop=True)

        casos = []
        for caso in zip(df[col_name_descripcion], df[col_name_responsabilidad], df[_id]):
            casos.append(Caso(caso[0], caso[1], caso[2]))
        return casos

    @classmethod
    def caso2csv(cls, casos, csv_name):
        """Guarda la lista de Sentencia en un csv

        :param cls: Sentencia, la clase
        :param sentencias: list, lista de sentencias Sentencia()
        :param csv_name: str, nombre del csv a guardar
        :returns: None

        """
        df = pd.DataFrame(columns=cls.__dict__.keys())
        for caso in casos:
            df = df.append(cls.__dict__)

        df.to_csv(csv_name)

    def set_clean_descripcion(self, clean_descripcion):
        self.clean_descripcion = clean_descripcion

    def get_clean_descripcion(self):
        return self.clean_descripcion

    def get_clean_descripcion(self):
        """Retorna la descripcion limpia

        :returns: str, descripcion limpia

        """
        self.set_clean_descripcion(
            clean.stem_string(
                clean.remove_stops(
                    clean.general(self.descripcion))))
        return self.clean_descripcion

    def set_resp_as_vector(self, traductor):
        """Solo cambia la responsabilidad a un vector binario o no dependiendo del
        traductor.

        :param traductor: dict, permite traducir la responsabilidad i.e.
        {'A':[1,0,0]}
        :returns: None

        """
        try:
            self.responsabilidad = traductor[self.responsabilidad]
        except KeyError:
            print('Clave no encontrada en el diccionario')

    def get_desc_vector(self, model):
        """Función que dado un vectorizador devuelve el vector de la descripcion

        :param model: tfidfVectorizer, vectorizador
        :returns: sparse_matrix, matriz sparse de la descripcion

        """
        if self.clean_descripcion:
            return model.transform([self.clean_descripcion])
        return model.tranform([self.get_clean_descripcion()])
