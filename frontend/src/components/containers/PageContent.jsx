const PageContent = ({ children, className }) => {
  return (
    <div
      className={`p-10 w-full ${className ? className : "flex flex-col items-center justify-start"} overflow-y-auto tracking-wide leading-8`}
    >
      {children}
    </div>
  );
};

export default PageContent;
