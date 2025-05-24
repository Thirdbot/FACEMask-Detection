import Tooltip from "@mui/material/Tooltip";
import { NavLink } from "react-router";
import InfoRoundedIcon from "@mui/icons-material/InfoRounded";
import TipsAndUpdatesRoundedIcon from "@mui/icons-material/TipsAndUpdatesRounded";
import List from "@mui/material/List";
import ListSubheader from "@mui/material/ListSubheader";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";
import HorizontalLine from "../ui/HorizontalLine";

const Manual = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="วิธีการใช้งาน" />
        <div className="w-full text-start tracking-wide">
          <div>
            <List
              subheader={
                <ListSubheader sx={{ fontSize: "1rem" }} color="primary">
                  <span>
                    <TipsAndUpdatesRoundedIcon className="me-1" />
                    ข้อควรปฏิบัติก่อนใช้งาน
                  </span>
                </ListSubheader>
              }
            >
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      1.) ให้แน่ใจว่าอุปกรณ์ของคุณมีกล้องอยู่ด้วยไม่งั้น AI
                      จะไม่สามารถทำนายผลลัพธ์ได้
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      2.)
                      ตรวจสอบให้แน่ใจว่าได้เชื่อมต่อกับสัญญาณอินเทอร์เน็ตหรือไม่
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      3.)
                      ตรวจสอบพื้นที่รอบข้างว่าที่ๆนั้นมีแสงสว่างเพียงพอหรือไม่
                      ไม่งั้น AI อาจจะทำนายผลลัพธ์ผิดได้
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      4.) AI
                      อาจจะทำนายค่าผลลัพธ์ผิดได้หากตรวจพบวัตถุอื่นๆเป็นจำนวนมากในวิดีโอให้ทำการเคลียร์วัตถุอื่นๆที่อยู่บริเวณรอบข้างของคุณเพื่อทำให้
                      AI ทำนายค่าได้แม่นยำขึ้น
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      5.) หากไม่ต้องการใช้งานแล้วให้คลิกปุ่มปิดกล้องทันที
                      ไม่งั้นตัว Server จะทำงานประมวลผลหนักเกินไปอาจทำให้ Server
                      ล้มได้
                    </span>
                  }
                />
              </ListItem>
            </List>
          </div>
          <HorizontalLine />
          <div className="mt-4">
            <List
              subheader={
                <ListSubheader sx={{ fontSize: "1rem" }} color="primary">
                  <span>
                    <InfoRoundedIcon className="me-1" />
                    ขั้นตอนการใช้งาน
                  </span>
                </ListSubheader>
              }
            >
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      1.) ให้กดไปที่หน้า
                      <Tooltip
                        title={<span>ไปยังหน้าตรวจสอบใบหน้า</span>}
                        placement="right"
                        arrow
                      >
                        <NavLink
                          to="/dashboard/face-mask-detection"
                          className="text-green-400 underline mx-1"
                        >
                          ตรวจสอบใบหน้า
                        </NavLink>
                      </Tooltip>
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText primary={<span>2.) กดที่ปุ่มเปิดกล้อง</span>} />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>
                      3.) รอกล้องเปิดแล้วตัว AI
                      จะทำการประมวลผลรูปภาพใบหน้าของเราว่าใส่หรือไม่ได้ใส่แมสโดยจะขึ้นเป็นกรอบสี่เหลี่ยมขึ้นมามารค์บนใบหน้าเรา
                    </span>
                  }
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <span>4.) หากต้องการหยุดใช้งานให้กดปุ่มปิดกล้อง</span>
                  }
                />
              </ListItem>
            </List>
          </div>
        </div>
      </PageContent>
    </AppContainer>
  );
};

export default Manual;
