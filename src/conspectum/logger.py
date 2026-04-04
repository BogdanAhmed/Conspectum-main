import enum
import typing


class Logger:
    class FileType(enum.Enum):
        TEXT = 0
        TEX = 1
        PDF = 2
        AUDIO = 3

        def get_extension(self):
            match self:
                case Logger.FileType.TEXT:
                    return "txt"
                case Logger.FileType.TEX:
                    return "tex"
                case Logger.FileType.PDF:
                    return "pdf"
                case Logger.FileType.AUDIO:
                    return "wav"
                case __:
                    assert False, f"Unsupported file type: {self}"

        def writing_mode(self):
            match self:
                case Logger.FileType.TEXT | Logger.FileType.TEX:
                    return "w"
                case Logger.FileType.PDF | Logger.FileType.AUDIO:
                    return "wb"
                case __:
                    assert False, f"Unsupported file type: {self}"

    def __init__(self, out_folder: typing.Optional[str] = None):
        self.out_folder_ = out_folder

    async def file(self, key: str, output: str, type: FileType):
        if not self.out_folder_:
            return
        with open(f"{self.out_folder_}/{key}.{type.get_extension()}", type.writing_mode()) as file:
            file.write(output)

    async def partial_result(self, text: str):
        print(text)

    async def progress(self, completed: int, total: int):
        await self.partial_result(f"I am {round(completed / total * 100)}% done")
