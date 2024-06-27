(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h c d)
(:init 
(handempty)
(ontable h)
(ontable c)
(ontable d)
(clear h)
(clear c)
(clear d)
)
(:goal
(and
(on h c)
(on c d)
)))