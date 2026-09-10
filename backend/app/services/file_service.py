from pathlib import Path


class FileService:
    """
    Service responsible for file-system operations
    related to uploaded documents.
    """

    def save_file(
        self,
        file_path: str,
        file_content: bytes,
    ) -> None:
        """
        Save uploaded file content to disk.
        """

        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "wb",
        ) as output_file:

            output_file.write(
                file_content
            )

    def delete_file(
        self,
        file_path: str,
    ) -> bool:
        """
        Delete a file if it exists.

        Returns:
            True  -> file was deleted
            False -> file did not exist
        """

        path = Path(file_path)

        if not path.exists():
            return False

        if not path.is_file():
            return False

        path.unlink()

        return True

    def file_exists(
        self,
        file_path: str,
    ) -> bool:
        """
        Check whether a file exists.
        """

        return Path(file_path).is_file()


file_service = FileService()