(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c g a d)
(:init 
(handempty)
(ontable c)
(ontable g)
(ontable a)
(ontable d)
(clear c)
(clear g)
(clear a)
(clear d)
)
(:goal
(and
(on c g)
(on g a)
(on a d)
)))