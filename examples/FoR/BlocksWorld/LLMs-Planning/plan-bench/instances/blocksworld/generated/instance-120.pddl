(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c j g f)
(:init 
(handempty)
(ontable c)
(ontable j)
(ontable g)
(ontable f)
(clear c)
(clear j)
(clear g)
(clear f)
)
(:goal
(and
(on c j)
(on j g)
(on g f)
)))