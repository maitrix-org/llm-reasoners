(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f c k h)
(:init 
(handempty)
(ontable f)
(ontable c)
(ontable k)
(ontable h)
(clear f)
(clear c)
(clear k)
(clear h)
)
(:goal
(and
(on f c)
(on c k)
(on k h)
)))