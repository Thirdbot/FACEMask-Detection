import React from "react";
import AppContainer from "../containers/AppContainer";
import Sidebar from "../ui/Sidebar";
import PageContent from "../containers/PageContent";
import Title from "../ui/Title";

const Home = () => {
  return (
    <AppContainer>
      <Sidebar />
      <PageContent>
        <Title text="หน้าหลัก" />
        <p>
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Excepturi
          voluptatibus incidunt corrupti illum harum vero. Exercitationem
          voluptatibus voluptates est corporis nulla consequuntur laboriosam
          ducimus ut, accusamus, saepe cupiditate, error harum?
        </p>
      </PageContent>
    </AppContainer>
  );
};

export default Home;
