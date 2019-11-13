
class Invite:

    def __init__(self):
        self.forum_id = None            # int32
        self.spectrum_id = None         # int32
        self.citizen_id = None          # int32
        self.handle = None              # String
        self.moniker = None             # String
        self.enlisted = None            # DateTime
        self.location = None            # Value Object: Location
        self.fluencies = None           # Value Object: FluencyContainer
        self.chat_last_presence = None  # DateTime
        self.has_custom_avatar = None   # Boolean
        self.invite_sent = None         # DateTime (today's date if running live)
        self.forum_last_active = None   # DateTime
        self.forum_visits = None        # int32
        self.recruit_status = None      # Boolean (true if joined)
