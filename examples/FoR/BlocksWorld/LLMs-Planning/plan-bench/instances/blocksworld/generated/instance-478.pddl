(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c h j)
(:init 
(handempty)
(ontable c)
(ontable h)
(ontable j)
(clear c)
(clear h)
(clear j)
)
(:goal
(and
(on c h)
(on h j)
)))