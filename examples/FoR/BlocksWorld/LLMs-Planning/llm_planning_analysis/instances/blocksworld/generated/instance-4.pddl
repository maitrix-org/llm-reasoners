(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g j f b)
(:init 
(handempty)
(ontable g)
(ontable j)
(ontable f)
(ontable b)
(clear g)
(clear j)
(clear f)
(clear b)
)
(:goal
(and
(on g j)
(on j f)
(on f b)
)))