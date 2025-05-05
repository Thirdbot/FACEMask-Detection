import { List, ListItem, ListItemText, Box } from "@mui/material";
import uuid from "react-uuid";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import Title from "../ui/Title";
import PageContent from "../containers/PageContent";
import { membersData } from "../constants";

const Members = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="รายชื่อสมาชิกในกลุ่ม" />
        <Box>
          <List>
            {membersData.map(({ name, studentId }, index) => (
              <ListItem key={uuid()}>
                <ListItemText>
                  <p className="text-center">
                    {index + 1}.) {name} รหัสนิสิต {studentId}
                  </p>
                </ListItemText>
              </ListItem>
            ))}
          </List>
        </Box>
      </PageContent>
    </AppContainer>
  );
};

export default Members;
