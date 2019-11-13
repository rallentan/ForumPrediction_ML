import pymysql


# Currently, the Spectrum Data Provider system only exposes data by direct access to its database.
from domain.entities.invite import Invite
from domain.value_objects.fluency_container import FluencyContainer
from domain.value_objects.location import Location
from domain.value_objects.recruit_status import RecruitStatus


class SpectrumDataProviderAdapter:

    def __init__(self, connection_info):
        self.__connection_info = connection_info

    def __enter__(self):
        self.connection = pymysql.connect(host=self.__connection_info['hostname'],
                                          port=self.__connection_info['port'],
                                          user=self.__connection_info['username'],
                                          password=self.__connection_info['password'],
                                          db=self.__connection_info['database'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def get_invites(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT * FROM `InviteHistory`")
            rows = cursor.fetchall()
            invites = []
            for row in rows:
                invite = Invite()
                invite.spectrum_id = row[1]
                invite.citizen_id = row[2]
                invite.handle = row[3]
                invite.moniker = row[4]
                invite.enlisted = row[5]
                invite.location = Location.from_string(row[6])
                invite.fluencies = FluencyContainer.from_comma_separated_string(row[7])
                invite.chat_last_presence = row[8]
                invite.has_custom_avatar = row[9]
                invite.invite_sent = row[10]
                invite.forum_last_active = row[11]
                invite.forum_visits = row[12]
                invite.recruit_status = RecruitStatus(row[13])
                invite.forum_id = row[14]
                invites.append(invite)
            return invites
        except Exception as ex:
            raise ex

    def get_countries(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT `CountryName` FROM `Countries` ORDER BY `CountryID`")
            rows = cursor.fetchall()
            countries = []
            for row in rows:
                countries.append(row[0])
            return countries
        except Exception as ex:
            raise ex

    def get_fluencies(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT `FluencyName` FROM `Fluencies` ORDER BY `FluencyID`")
            rows = cursor.fetchall()
            fluencies = []
            for row in rows:
                fluencies.append(row[0])
            return fluencies
        except Exception as ex:
            raise ex
