(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g d f b)
(:init 
(handempty)
(ontable g)
(ontable d)
(ontable f)
(ontable b)
(clear g)
(clear d)
(clear f)
(clear b)
)
(:goal
(and
(on g d)
(on d f)
(on f b)
)))