import { List, ListItem, ListItemText } from "@mui/material";
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
        <div className="w-full overflow-hidden">
          <List>
            {membersData.map(({ name, studentId, responsibility }, index) => (
              <ListItem key={uuid()}>
                <ListItemText>
                  <p className="text-start">
                    {index + 1}.) {name} รหัสนิสิต {studentId} รับผิดชอบหน้าที่ {responsibility}
                  </p>
                </ListItemText>
              </ListItem>
            ))}
          </List>
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default Members;
