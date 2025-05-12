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
        <div className="w-full overflow-hidden flex items-center justify-evenly flex-wrap">
          <List>
            <ListItem>
              <ListItemText>
                <p className="text-center text-lg underline">ชื่อสมาชิก</p>
              </ListItemText>
            </ListItem>
            {membersData.map(({ name, studentId }, index) => (
              <ListItem key={uuid()}>
                <ListItemText>
                  <p className="text-start">
                    {index + 1}.) {name} รหัสนิสิต {studentId}
                  </p>
                </ListItemText>
              </ListItem>
            ))}
          </List>
          <List>
            <ListItem>
              <ListItemText>
                <p className="text-center text-lg underline">บทบาทหน้าที่ๆรับผิดชอบ</p>
              </ListItemText>
            </ListItem>
            {membersData.map(({ responsibility }) => (
              <ListItem key={uuid()}>
                <ListItemText>
                  <p className="text-start">
                    รับผิดชอบ {!!responsibility ? responsibility : "..."}
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
